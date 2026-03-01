#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace py = pybind11;

// ═══════════════════════════════════════════════════════════════════════════════
// Op types
// ═══════════════════════════════════════════════════════════════════════════════

struct CliffordOp {
    std::string name;
    std::vector<int> targets;

    CliffordOp(std::string name, std::vector<int> targets)
        : name(std::move(name)), targets(std::move(targets)) {}

    void apply(py::object sim) const {
        py::tuple args(targets.size());
        for (size_t i = 0; i < targets.size(); i++)
            args[i] = py::cast(targets[i]);

        if      (name == "H")            sim.attr("h")(*args);
        else if (name == "H_YZ")         sim.attr("h_yz")(*args);
        else if (name == "H_XY")         sim.attr("h_xy")(*args);
        else if (name == "S")            sim.attr("s")(*args);
        else if (name == "Sdg")          sim.attr("s_dag")(*args);
        else if (name == "SQRT_X")       sim.attr("sqrt_x")(*args);
        else if (name == "SQRT_X_DAG")   sim.attr("sqrt_x_dag")(*args);
        else if (name == "SQRT_Y")       sim.attr("sqrt_y")(*args);
        else if (name == "SQRT_Y_DAG")   sim.attr("sqrt_y_dag")(*args);
        else if (name == "X")            sim.attr("x")(*args);
        else if (name == "Y")            sim.attr("y")(*args);
        else if (name == "Z")            sim.attr("z")(*args);
        else if (name == "CX")           sim.attr("cx")(*args);
        else if (name == "CZ")           sim.attr("cz")(*args);
        else if (name == "CY")           sim.attr("cy")(*args);
        else if (name == "SWAP")         sim.attr("swap")(*args);
        else if (name == "ISWAP")        sim.attr("iswap")(*args);
        else if (name == "ISWAP_DAG")    sim.attr("iswap_dag")(*args);
        else if (name == "R")            sim.attr("reset")(*args);
        else if (name == "RX")           sim.attr("reset_x")(*args);
        else if (name == "RY")           sim.attr("reset_y")(*args);
        else if (name == "I")            { /* noop */ }
        else throw std::runtime_error("Unknown Clifford gate: '" + name + "'");
    }
};

struct RZOp {
    std::vector<int> qubits;
    double theta;

    RZOp(std::vector<int> qubits, double theta)
        : qubits(std::move(qubits)), theta(theta) {}
};

struct PauliNoiseOp {
    std::vector<int> qubits;
    double px, py, pz;

    PauliNoiseOp(std::vector<int> qubits, double px, double py, double pz)
        : qubits(std::move(qubits)), px(px), py(py), pz(pz) {}

    double p_identity() const { return 1.0 - px - py - pz; }
};

struct TwoQubitPauliNoiseOp {
    int q0, q1;
    std::vector<double> probs;

    TwoQubitPauliNoiseOp(int q0, int q1, std::vector<double> probs)
        : q0(q0), q1(q1), probs(std::move(probs)) {}

    double p_identity() const {
        return 1.0 - std::accumulate(probs.begin(), probs.end(), 0.0);
    }
};

struct TickOp {};

struct DampOp {
    std::vector<int> qubits;
    double t, T1, T2;

    DampOp(std::vector<int> qubits, double t, double T1, double T2)
        : qubits(std::move(qubits)), t(t), T1(T1), T2(T2) {}
};

struct MeasureOp {
    std::vector<int> qubits;
    std::string basis;
    bool reset;
    double flip_probability;

    MeasureOp(std::vector<int> qubits, std::string basis = "Z",
              bool reset = false, double flip_probability = 0.0)
        : qubits(std::move(qubits)), basis(std::move(basis)),
          reset(reset), flip_probability(flip_probability) {}

    bool apply(py::object sim) const {
        int max_q = *std::max_element(qubits.begin(), qubits.end());
        py::module_ stim_mod = py::module_::import("stim");
        py::object ps = stim_mod.attr("PauliString")(max_q + 1);
        for (int q : qubits)
            ps[py::cast(q)] = py::cast(basis);

        bool result = sim.attr("measure_observable")(
            ps, py::arg("flip_probability") = flip_probability
        ).cast<bool>();

        if (reset) {
            py::tuple args(qubits.size());
            for (size_t i = 0; i < qubits.size(); i++)
                args[i] = py::cast(qubits[i]);
            if      (basis == "Z") sim.attr("reset")(*args);
            else if (basis == "X") sim.attr("reset_x")(*args);
            else if (basis == "Y") sim.attr("reset_y")(*args);
        }
        return result;
    }
};

using OpVariant = std::variant<
    CliffordOp, RZOp, PauliNoiseOp, TwoQubitPauliNoiseOp,
    TickOp, DampOp, MeasureOp>;

// ═══════════════════════════════════════════════════════════════════════════════
// Gate-name lookup tables (matching the Python _STIM_*_CLIFFORD dicts)
// ═══════════════════════════════════════════════════════════════════════════════

static const std::unordered_map<std::string, std::string> STIM_1Q_CLIFFORD = {
    {"H","H"}, {"H_YZ","H_YZ"}, {"H_XY","H_XY"}, {"I","I"},
    {"S","S"}, {"SQRT_Z","S"}, {"S_DAG","Sdg"}, {"SQRT_Z_DAG","Sdg"},
    {"X","X"}, {"Y","Y"}, {"Z","Z"},
    {"SQRT_X","SQRT_X"}, {"SX","SQRT_X"},
    {"SQRT_X_DAG","SQRT_X_DAG"}, {"SX_DAG","SQRT_X_DAG"},
    {"SQRT_Y","SQRT_Y"}, {"SQRT_Y_DAG","SQRT_Y_DAG"},
    {"R","R"}, {"RX","RX"}, {"RY","RY"},
};

static const std::unordered_map<std::string, std::string> STIM_2Q_CLIFFORD = {
    {"CX","CX"}, {"CNOT","CX"}, {"ZCX","CX"},
    {"CY","CY"}, {"CZ","CZ"}, {"ZCZ","CZ"},
    {"SWAP","SWAP"}, {"ISWAP","ISWAP"}, {"ISWAP_DAG","ISWAP_DAG"},
};

static const std::unordered_set<std::string> STIM_SKIP = {
    "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS",
    "MPP", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1", "MPAD",
};

struct MeasInfo { std::string basis; bool reset; };
static const std::unordered_map<std::string, MeasInfo> MEAS_MAP = {
    {"M",{"Z",false}},  {"MZ",{"Z",false}},
    {"MR",{"Z",true}},  {"MRZ",{"Z",true}},
    {"MX",{"X",false}}, {"MRX",{"X",true}},
    {"MY",{"Y",false}}, {"MRY",{"Y",true}},
};

// ═══════════════════════════════════════════════════════════════════════════════
// op_to_str (Stim-like text format)
// ═══════════════════════════════════════════════════════════════════════════════

static std::string fmt(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

static std::string op_to_str(const OpVariant& op) {
    return std::visit([](const auto& o) -> std::string {
        using T = std::decay_t<decltype(o)>;

        if constexpr (std::is_same_v<T, TickOp>) {
            return "TICK";
        }
        else if constexpr (std::is_same_v<T, CliffordOp>) {
            std::string n = (o.name == "Sdg") ? "S_DAG" : o.name;
            std::ostringstream ss;
            ss << n;
            for (auto t : o.targets) ss << ' ' << t;
            return ss.str();
        }
        else if constexpr (std::is_same_v<T, RZOp>) {
            std::ostringstream ss;
            ss << "RZ(" << fmt(o.theta) << ')';
            for (auto q : o.qubits) ss << ' ' << q;
            return ss.str();
        }
        else if constexpr (std::is_same_v<T, PauliNoiseOp>) {
            std::ostringstream ss;
            if (std::abs(o.px - o.py) < 1e-12 && std::abs(o.py - o.pz) < 1e-12)
                ss << "DEPOLARIZE1(" << fmt(o.px * 3) << ')';
            else
                ss << "PAULI_CHANNEL_1(" << fmt(o.px) << ',' << fmt(o.py) << ',' << fmt(o.pz) << ')';
            for (auto q : o.qubits) ss << ' ' << q;
            return ss.str();
        }
        else if constexpr (std::is_same_v<T, TwoQubitPauliNoiseOp>) {
            double p0 = o.probs[0];
            bool all_same = std::all_of(o.probs.begin(), o.probs.end(),
                                        [p0](double p){ return std::abs(p - p0) < 1e-12; });
            std::ostringstream ss;
            if (all_same) {
                ss << "DEPOLARIZE2(" << fmt(p0 * 15) << ')';
            } else {
                ss << "PAULI_CHANNEL_2(";
                for (size_t i = 0; i < o.probs.size(); i++) {
                    if (i) ss << ',';
                    ss << fmt(o.probs[i]);
                }
                ss << ')';
            }
            ss << ' ' << o.q0 << ' ' << o.q1;
            return ss.str();
        }
        else if constexpr (std::is_same_v<T, DampOp>) {
            std::ostringstream ss;
            std::string t1s = std::isinf(o.T1) ? "inf" : fmt(o.T1);
            std::string t2s = std::isinf(o.T2) ? "inf" : fmt(o.T2);
            ss << "DAMP(t=" << fmt(o.t) << ",T1=" << t1s << ",T2=" << t2s << ')';
            for (auto q : o.qubits) ss << ' ' << q;
            return ss.str();
        }
        else if constexpr (std::is_same_v<T, MeasureOp>) {
            std::ostringstream ss;
            std::string bsuf = (o.basis == "Z") ? "" : o.basis;
            std::string n = (o.reset ? "MR" : "M") + bsuf;
            if (o.flip_probability > 0)
                ss << n << '(' << fmt(o.flip_probability) << ')';
            else
                ss << n;
            for (auto q : o.qubits) ss << ' ' << q;
            return ss.str();
        }
        return "UNKNOWN";
    }, op);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Circuit class
// ═══════════════════════════════════════════════════════════════════════════════

class Circuit {
public:
    int n_qubits_;
    std::vector<OpVariant> ops_;

    explicit Circuit(int n_qubits) : n_qubits_(n_qubits) {
        if (n_qubits < 1)
            throw std::invalid_argument("n_qubits must be >= 1");
    }

    // -- helpers ----------------------------------------------------------

    static std::vector<int> to_qubits(py::object q) {
        if (py::isinstance<py::int_>(q))
            return {q.cast<int>()};
        std::vector<int> out;
        for (auto item : q)
            out.push_back(item.cast<int>());
        return out;
    }

    // -- Clifford builders ------------------------------------------------

    Circuit& h   (py::args a) { return add_cliff("H", a); }
    Circuit& s   (py::args a) { return add_cliff("S", a); }
    Circuit& sdg (py::args a) { return add_cliff("Sdg", a); }
    Circuit& x   (py::args a) { return add_cliff("X", a); }
    Circuit& y   (py::args a) { return add_cliff("Y", a); }
    Circuit& z   (py::args a) { return add_cliff("Z", a); }
    Circuit& cx  (py::args a) { return add_cliff("CX", a); }
    Circuit& cz  (py::args a) { return add_cliff("CZ", a); }
    Circuit& cy  (py::args a) { return add_cliff("CY", a); }
    Circuit& do_swap(py::args a) { return add_cliff("SWAP", a); }
    Circuit& do_reset(py::args a) { return add_cliff("R", a); }

    // -- Measurement builders ---------------------------------------------

    Circuit& measure        (py::args a) { return add_meas(a, "Z", false); }
    Circuit& measure_reset  (py::args a) { return add_meas(a, "Z", true); }
    Circuit& measure_x      (py::args a) { return add_meas(a, "X", false); }
    Circuit& measure_reset_x(py::args a) { return add_meas(a, "X", true); }
    Circuit& measure_y      (py::args a) { return add_meas(a, "Y", false); }
    Circuit& measure_reset_y(py::args a) { return add_meas(a, "Y", true); }

    Circuit& tick() { ops_.emplace_back(TickOp{}); return *this; }

    // -- RZ ---------------------------------------------------------------

    Circuit& rz(py::object qubits, double theta) {
        ops_.emplace_back(RZOp(to_qubits(qubits), theta));
        return *this;
    }

    // -- Noise builders ---------------------------------------------------

    Circuit& pauli_channel_1(py::object qubits,
                             double px, double py, double pz) {
        ops_.emplace_back(PauliNoiseOp(to_qubits(qubits), px, py, pz));
        return *this;
    }
    Circuit& depolarize1(py::object qubits, double p) {
        return pauli_channel_1(qubits, p/3, p/3, p/3);
    }
    Circuit& x_error(py::object qubits, double p) {
        return pauli_channel_1(qubits, p, 0.0, 0.0);
    }
    Circuit& y_error(py::object qubits, double p) {
        return pauli_channel_1(qubits, 0.0, p, 0.0);
    }
    Circuit& z_error(py::object qubits, double p) {
        return pauli_channel_1(qubits, 0.0, 0.0, p);
    }
    Circuit& pauli_channel_2(int q0, int q1, std::vector<double> probs) {
        if (probs.size() != 15)
            throw std::invalid_argument(
                "Expected 15 probabilities, got " + std::to_string(probs.size()));
        ops_.emplace_back(TwoQubitPauliNoiseOp(q0, q1, std::move(probs)));
        return *this;
    }
    Circuit& depolarize2(int q0, int q1, double p) {
        std::vector<double> probs(15, p / 15.0);
        return pauli_channel_2(q0, q1, std::move(probs));
    }
    Circuit& damp(py::object qubits, double t, double T1, double T2) {
        ops_.emplace_back(DampOp(to_qubits(qubits), t, T1, T2));
        return *this;
    }

    // -- idle-noise insertion ---------------------------------------------

    Circuit add_idle_exact_before_measurement(
            double t, double T1, double T2) const {
        Circuit nc(n_qubits_);
        std::vector<int> all(n_qubits_);
        std::iota(all.begin(), all.end(), 0);

        bool prev_meas = false;
        for (const auto& op : ops_) {
            if (std::holds_alternative<MeasureOp>(op)) {
                if (!prev_meas)
                    nc.ops_.emplace_back(DampOp(all, t, T1, T2));
                prev_meas = true;
            } else {
                prev_meas = false;
            }
            nc.ops_.push_back(op);
        }
        return nc;
    }

    // -- from_stim --------------------------------------------------------

    static Circuit from_stim(py::object stim_circuit,
                             py::object n_qubits_opt = py::none()) {
        int nq;
        if (n_qubits_opt.is_none()) {
            nq = stim_circuit.attr("num_qubits").cast<int>();
            if (nq == 0) nq = 1;
        } else {
            nq = n_qubits_opt.cast<int>();
        }
        Circuit c(nq);
        c.extend_from_stim(stim_circuit);
        return c;
    }

    void extend_from_stim(py::object sc) {
        py::module_ stim_mod = py::module_::import("stim");
        py::object RepeatBlock = stim_mod.attr("CircuitRepeatBlock");

        for (auto handle : sc) {
            py::object instr = py::reinterpret_borrow<py::object>(handle);

            // REPEAT blocks: unroll
            if (py::isinstance(instr, RepeatBlock)) {
                int cnt = instr.attr("repeat_count").cast<int>();
                py::object body = instr.attr("body_copy")();
                for (int i = 0; i < cnt; i++)
                    extend_from_stim(body);
                continue;
            }

            std::string iname = instr.attr("name").cast<std::string>();
            py::list tgt_py   = instr.attr("targets_copy")();
            py::list args_py  = instr.attr("gate_args_copy")();

            std::vector<int> tgts;
            for (auto t : tgt_py)
                if (t.attr("is_qubit_target").cast<bool>())
                    tgts.push_back(t.attr("value").cast<int>());

            std::vector<double> args;
            for (auto a : args_py)
                args.push_back(a.cast<double>());

            // 1Q Clifford
            {auto it = STIM_1Q_CLIFFORD.find(iname);
             if (it != STIM_1Q_CLIFFORD.end()) {
                ops_.emplace_back(CliffordOp(it->second, tgts)); continue; }}

            // 2Q Clifford
            {auto it = STIM_2Q_CLIFFORD.find(iname);
             if (it != STIM_2Q_CLIFFORD.end()) {
                ops_.emplace_back(CliffordOp(it->second, tgts)); continue; }}

            // Noise channels
            if (iname == "PAULI_CHANNEL_1") {
                ops_.emplace_back(PauliNoiseOp(tgts, args[0], args[1], args[2]));
            }
            else if (iname == "DEPOLARIZE1") {
                double p = args[0];
                ops_.emplace_back(PauliNoiseOp(tgts, p/3, p/3, p/3));
            }
            else if (iname == "X_ERROR" || iname == "Y_ERROR" || iname == "Z_ERROR") {
                double p = args[0];
                ops_.emplace_back(PauliNoiseOp(tgts,
                    iname == "X_ERROR" ? p : 0.0,
                    iname == "Y_ERROR" ? p : 0.0,
                    iname == "Z_ERROR" ? p : 0.0));
            }
            else if (iname == "PAULI_CHANNEL_2") {
                std::vector<double> pr(args.begin(), args.end());
                for (size_t i = 0; i + 1 < tgts.size(); i += 2)
                    ops_.emplace_back(TwoQubitPauliNoiseOp(tgts[i], tgts[i+1], pr));
            }
            else if (iname == "DEPOLARIZE2") {
                std::vector<double> pr(15, args[0] / 15.0);
                for (size_t i = 0; i + 1 < tgts.size(); i += 2)
                    ops_.emplace_back(TwoQubitPauliNoiseOp(tgts[i], tgts[i+1], pr));
            }
            // Measurements
            else if (MEAS_MAP.count(iname)) {
                const auto& mi = MEAS_MAP.at(iname);
                double fp = args.empty() ? 0.0 : args[0];
                for (int t : tgts)
                    ops_.emplace_back(MeasureOp({t}, mi.basis, mi.reset, fp));
            }
            // Tick
            else if (iname == "TICK") {
                ops_.emplace_back(TickOp{});
            }
            // Skip
            else if (STIM_SKIP.count(iname)) { /* silent */ }
            else {
                py::module_::import("warnings").attr("warn")(
                    "Circuit.from_stim: unsupported instruction '" + iname + "' skipped.");
            }
        }
    }

    // -- inspection -------------------------------------------------------

    py::list get_ops() const {
        py::list out;
        for (const auto& op : ops_)
            std::visit([&](const auto& o){ out.append(py::cast(o)); }, op);
        return out;
    }

    int n_clifford() const {
        int c = 0;
        for (auto& o : ops_) c += std::holds_alternative<CliffordOp>(o);
        return c;
    }
    int n_rz() const {
        int c = 0;
        for (auto& o : ops_) c += std::holds_alternative<RZOp>(o);
        return c;
    }
    int n_noise() const {
        int c = 0;
        for (auto& o : ops_) {
            c += std::holds_alternative<PauliNoiseOp>(o);
            c += std::holds_alternative<TwoQubitPauliNoiseOp>(o);
        }
        return c;
    }
    int n_meas() const {
        int c = 0;
        for (auto& o : ops_) c += std::holds_alternative<MeasureOp>(o);
        return c;
    }

    size_t size() const { return ops_.size(); }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "Circuit(n_qubits=" << n_qubits_
           << ", clifford=" << n_clifford()
           << ", rz=" << n_rz()
           << ", noise=" << n_noise()
           << ", measurements=" << n_meas() << ')';
        for (const auto& op : ops_)
            ss << '\n' << op_to_str(op);
        return ss.str();
    }

private:
    Circuit& add_cliff(const std::string& gate, py::args a) {
        std::vector<int> qs;
        qs.reserve(a.size());
        for (auto q : a) qs.push_back(q.cast<int>());
        ops_.emplace_back(CliffordOp(gate, std::move(qs)));
        return *this;
    }
    Circuit& add_meas(py::args a, const std::string& basis, bool rst) {
        for (auto q : a)
            ops_.emplace_back(MeasureOp({q.cast<int>()}, basis, rst, 0.0));
        return *this;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// pybind11 module
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_circuit, m) {
    m.doc() = "C++ implementation of QC_claude circuit types";

    // -- CliffordOp -------------------------------------------------------
    py::class_<CliffordOp>(m, "CliffordOp")
        .def(py::init([](std::string name, py::object targets) {
            std::vector<int> v;
            for (auto t : targets) v.push_back(t.cast<int>());
            return CliffordOp(std::move(name), std::move(v));
        }), py::arg("name"), py::arg("targets"))
        .def_readonly("name", &CliffordOp::name)
        .def_property_readonly("targets", [](const CliffordOp& s) {
            return py::tuple(py::cast(s.targets));
        })
        .def("apply", &CliffordOp::apply, py::arg("sim"))
        .def("__repr__", [](const CliffordOp& s) {
            std::ostringstream ss;
            ss << "CliffordOp('" << s.name << "', (";
            for (size_t i = 0; i < s.targets.size(); i++) {
                if (i) ss << ", ";
                ss << s.targets[i];
            }
            if (s.targets.size() == 1) ss << ',';
            ss << "))";
            return ss.str();
        });

    // -- RZOp -------------------------------------------------------------
    py::class_<RZOp>(m, "RZOp")
        .def(py::init([](py::object qubits, double theta) {
            std::vector<int> v;
            if (py::isinstance<py::int_>(qubits)) v.push_back(qubits.cast<int>());
            else for (auto q : qubits) v.push_back(q.cast<int>());
            return RZOp(std::move(v), theta);
        }), py::arg("qubits"), py::arg("theta"))
        .def_property_readonly("qubits", [](const RZOp& s) {
            return py::tuple(py::cast(s.qubits));
        })
        .def_readonly("theta", &RZOp::theta);

    // -- PauliNoiseOp -----------------------------------------------------
    py::class_<PauliNoiseOp>(m, "PauliNoiseOp")
        .def(py::init([](py::object qubits, double px, double py, double pz) {
            std::vector<int> v;
            if (py::isinstance<py::int_>(qubits)) v.push_back(qubits.cast<int>());
            else for (auto q : qubits) v.push_back(q.cast<int>());
            return PauliNoiseOp(std::move(v), px, py, pz);
        }), py::arg("qubits"), py::arg("px"), py::arg("py"), py::arg("pz"))
        .def_property_readonly("qubits", [](const PauliNoiseOp& s) {
            return py::tuple(py::cast(s.qubits));
        })
        .def_readonly("px", &PauliNoiseOp::px)
        .def_readonly("py", &PauliNoiseOp::py)
        .def_readonly("pz", &PauliNoiseOp::pz)
        .def_property_readonly("p_identity", &PauliNoiseOp::p_identity);

    // -- TwoQubitPauliNoiseOp ---------------------------------------------
    py::class_<TwoQubitPauliNoiseOp>(m, "TwoQubitPauliNoiseOp")
        .def(py::init<int, int, std::vector<double>>(),
             py::arg("q0"), py::arg("q1"), py::arg("probs"))
        .def_readonly("q0", &TwoQubitPauliNoiseOp::q0)
        .def_readonly("q1", &TwoQubitPauliNoiseOp::q1)
        .def_property_readonly("probs", [](const TwoQubitPauliNoiseOp& s) {
            return py::tuple(py::cast(s.probs));
        })
        .def_property_readonly("p_identity", &TwoQubitPauliNoiseOp::p_identity);

    // -- TickOp -----------------------------------------------------------
    py::class_<TickOp>(m, "TickOp")
        .def(py::init<>());

    // -- DampOp -----------------------------------------------------------
    py::class_<DampOp>(m, "DampOp")
        .def(py::init([](py::object qubits, double t, double T1, double T2) {
            std::vector<int> v;
            if (py::isinstance<py::int_>(qubits)) v.push_back(qubits.cast<int>());
            else for (auto q : qubits) v.push_back(q.cast<int>());
            return DampOp(std::move(v), t, T1, T2);
        }), py::arg("qubits"), py::arg("t"), py::arg("T1"), py::arg("T2"))
        .def_property_readonly("qubits", [](const DampOp& s) {
            return py::tuple(py::cast(s.qubits));
        })
        .def_readonly("t", &DampOp::t)
        .def_readonly("T1", &DampOp::T1)
        .def_readonly("T2", &DampOp::T2);

    // -- MeasureOp --------------------------------------------------------
    py::class_<MeasureOp>(m, "MeasureOp")
        .def(py::init([](py::object qubits, std::string basis,
                         bool reset, double flip_probability) {
            std::vector<int> v;
            if (py::isinstance<py::int_>(qubits)) v.push_back(qubits.cast<int>());
            else for (auto q : qubits) v.push_back(q.cast<int>());
            return MeasureOp(std::move(v), std::move(basis), reset, flip_probability);
        }), py::arg("qubits"), py::arg("basis") = "Z",
            py::arg("reset") = false, py::arg("flip_probability") = 0.0)
        .def_property_readonly("qubits", [](const MeasureOp& s) {
            return py::tuple(py::cast(s.qubits));
        })
        .def_readonly("basis", &MeasureOp::basis)
        .def_readonly("reset", &MeasureOp::reset)
        .def_readonly("flip_probability", &MeasureOp::flip_probability)
        .def("apply", &MeasureOp::apply, py::arg("sim"));

    // -- Circuit ----------------------------------------------------------
    auto circ = py::class_<Circuit>(m, "Circuit");
    circ
        .def(py::init<int>(), py::arg("n_qubits"))
        .def_readwrite("n_qubits", &Circuit::n_qubits_)

        // Clifford builders
        .def("h",   &Circuit::h,   py::return_value_policy::reference_internal)
        .def("s",   &Circuit::s,   py::return_value_policy::reference_internal)
        .def("sdg", &Circuit::sdg, py::return_value_policy::reference_internal)
        .def("x",   &Circuit::x,   py::return_value_policy::reference_internal)
        .def("y",   &Circuit::y,   py::return_value_policy::reference_internal)
        .def("z",   &Circuit::z,   py::return_value_policy::reference_internal)
        .def("cx",  &Circuit::cx,  py::return_value_policy::reference_internal)
        .def("cz",  &Circuit::cz,  py::return_value_policy::reference_internal)
        .def("cy",  &Circuit::cy,  py::return_value_policy::reference_internal)
        .def("swap",  &Circuit::do_swap,  py::return_value_policy::reference_internal)
        .def("reset", &Circuit::do_reset, py::return_value_policy::reference_internal)

        // Measurement builders
        .def("measure",         &Circuit::measure,         py::return_value_policy::reference_internal)
        .def("measure_reset",   &Circuit::measure_reset,   py::return_value_policy::reference_internal)
        .def("measure_x",       &Circuit::measure_x,       py::return_value_policy::reference_internal)
        .def("measure_reset_x", &Circuit::measure_reset_x, py::return_value_policy::reference_internal)
        .def("measure_y",       &Circuit::measure_y,       py::return_value_policy::reference_internal)
        .def("measure_reset_y", &Circuit::measure_reset_y, py::return_value_policy::reference_internal)

        .def("tick", &Circuit::tick, py::return_value_policy::reference_internal)

        // RZ
        .def("rz", &Circuit::rz,
             py::arg("qubits"), py::arg("theta"),
             py::return_value_policy::reference_internal)

        // Noise builders
        .def("pauli_channel_1", &Circuit::pauli_channel_1,
             py::arg("qubits"), py::arg("px"), py::arg("py"), py::arg("pz"),
             py::return_value_policy::reference_internal)
        .def("depolarize1", &Circuit::depolarize1,
             py::arg("qubits"), py::arg("p"),
             py::return_value_policy::reference_internal)
        .def("x_error", &Circuit::x_error,
             py::arg("qubits"), py::arg("p"),
             py::return_value_policy::reference_internal)
        .def("y_error", &Circuit::y_error,
             py::arg("qubits"), py::arg("p"),
             py::return_value_policy::reference_internal)
        .def("z_error", &Circuit::z_error,
             py::arg("qubits"), py::arg("p"),
             py::return_value_policy::reference_internal)
        .def("pauli_channel_2", &Circuit::pauli_channel_2,
             py::arg("q0"), py::arg("q1"), py::arg("probs"),
             py::return_value_policy::reference_internal)
        .def("depolarize2", &Circuit::depolarize2,
             py::arg("q0"), py::arg("q1"), py::arg("p"),
             py::return_value_policy::reference_internal)
        .def("damp", &Circuit::damp,
             py::arg("qubits"), py::arg("t"), py::arg("T1"), py::arg("T2"),
             py::return_value_policy::reference_internal)

        // Idle noise insertion
        .def("add_idle_exact_before_measurement",
             &Circuit::add_idle_exact_before_measurement,
             py::arg("t"), py::arg("T1"), py::arg("T2"))

        // from_stim
        .def_static("from_stim", &Circuit::from_stim,
                     py::arg("stim_circuit"), py::arg("n_qubits") = py::none())

        // Properties
        .def_property_readonly("ops",            &Circuit::get_ops)
        .def_property_readonly("n_clifford_gates", &Circuit::n_clifford)
        .def_property_readonly("n_rz_gates",     &Circuit::n_rz)
        .def_property_readonly("n_noise_ops",    &Circuit::n_noise)
        .def_property_readonly("n_measurements", &Circuit::n_meas)

        .def("__len__",  &Circuit::size)
        .def("__str__",  &Circuit::to_string)
        .def("__repr__", &Circuit::to_string)
    ;
}
