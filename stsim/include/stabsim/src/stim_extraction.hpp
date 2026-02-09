#pragma once

#include "../../circuit.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace NWQSim
{
    namespace detail
    {
        struct StimLine
        {
            std::string text;
            int line_number;
        };

        inline std::string trim(const std::string &value)
        {
            const auto begin = value.find_first_not_of(" \t\r\n");
            if (begin == std::string::npos)
            {
                return "";
            }
            const auto end = value.find_last_not_of(" \t\r\n");
            return value.substr(begin, end - begin + 1);
        }

        inline std::string strip_comment(const std::string &value)
        {
            const auto hash = value.find('#');
            if (hash == std::string::npos)
            {
                return value;
            }
            return value.substr(0, hash);
        }

        inline std::vector<std::string> tokenize(const std::string &line)
        {
            std::vector<std::string> tokens;
            std::string current;
            int paren_depth = 0;
            
            for (size_t i = 0; i < line.size(); ++i)
            {
                char ch = line[i];
                if (ch == '(')
                {
                    paren_depth++;
                    current += ch;
                }
                else if (ch == ')')
                {
                    paren_depth--;
                    current += ch;
                }
                else if ((ch == ' ' || ch == '\t') && paren_depth == 0)
                {
                    if (!current.empty())
                    {
                        tokens.push_back(current);
                        current.clear();
                    }
                }
                else if (ch != ' ' && ch != '\t')
                {
                    current += ch;
                }
                else if (paren_depth > 0)
                {
                    // Skip spaces inside parentheses (e.g., "PAULI_CHANNEL_1(0.01, 0.01, 0.01)")
                    // Don't add spaces to the token
                }
            }
            if (!current.empty())
            {
                tokens.push_back(current);
            }
            return tokens;
        }

        inline std::string to_upper(std::string value)
        {
            for (char &ch : value)
            {
                ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            }
            return value;
        }

        // Parse instruction name and parameters from tokens like "DEPOLARIZE1(0.001)" or "PAULI_CHANNEL_1(0.01, 0.01, 0.01)"
        inline bool parse_parameterized_op(const std::string &token,
                                           std::string &op_name,
                                           std::vector<double> &params)
        {
            auto paren_open = token.find('(');
            if (paren_open == std::string::npos)
            {
                op_name = token;
                params.clear();
                return true;
            }
            auto paren_close = token.rfind(')');
            if (paren_close == std::string::npos || paren_close <= paren_open)
            {
                return false;
            }
            op_name = token.substr(0, paren_open);
            std::string param_str = token.substr(paren_open + 1, paren_close - paren_open - 1);
            params.clear();
            std::istringstream param_stream(param_str);
            std::string param_token;
            while (std::getline(param_stream, param_token, ','))
            {
                param_token = trim(param_token);
                if (!param_token.empty())
                {
                    params.push_back(std::stod(param_token));
                }
            }
            return true;
        }

        class StimParser
        {
        public:
            StimParser(Circuit &circuit, int requested_qubits)
                : circuit_(circuit)
            {
                IdxType existing = circuit_.num_qubits();
                if (existing > 0)
                {
                    max_qubit_seen_ = existing - 1;
                }
                if (requested_qubits > 0)
                {
                    max_qubit_seen_ = std::max<IdxType>(max_qubit_seen_, static_cast<IdxType>(requested_qubits - 1));
                }
                final_qubits_ = circuit_.num_qubits();
            }

            void parse_from_string(const std::string &source)
            {
                std::istringstream input(source);
                std::string line;
                int line_number = 0;
                std::vector<StimLine> lines;
                while (std::getline(input, line))
                {
                    lines.push_back({line, ++line_number});
                }
                size_t idx = 0;
                process_block(lines, idx);
                finalize();
            }

            IdxType final_qubit_count() const
            {
                return final_qubits_;
            }

        private:
            Circuit &circuit_;
            IdxType max_qubit_seen_ = -1;
            IdxType final_qubits_ = 0;

            void finalize()
            {
                if (max_qubit_seen_ < 0)
                {
                    final_qubits_ = circuit_.num_qubits();
                    return;
                }
                IdxType desired = max_qubit_seen_ + 1;
                if (desired > circuit_.num_qubits())
                {
                    circuit_.set_num_qubits(desired);
                }
                final_qubits_ = circuit_.num_qubits();
            }

            void process_block(const std::vector<StimLine> &lines, size_t &idx)
            {
                while (idx < lines.size())
                {
                    const StimLine &line = lines[idx++];
                    std::string sanitized = trim(strip_comment(line.text));
                    if (sanitized.empty())
                    {
                        continue;
                    }
                    if (sanitized == "}")
                    {
                        return;
                    }
                    auto tokens = tokenize(sanitized);
                    if (tokens.empty())
                    {
                        continue;
                    }
                    std::string op_upper = to_upper(tokens[0]);
                    if (op_upper == "REPEAT")
                    {
                        handle_repeat(line, tokens, lines, idx);
                        continue;
                    }
                    handle_instruction(line, op_upper, tokens);
                }
            }

            void handle_repeat(const StimLine &line,
                               const std::vector<std::string> &tokens,
                               const std::vector<StimLine> &lines,
                               size_t &idx)
            {
                if (tokens.size() < 3 || tokens[2] != "{")
                {
                    throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): expected '{' after REPEAT count");
                }
                int repetitions = std::stoi(tokens[1]);
                auto block = collect_block(lines, idx, line.line_number);
                for (int rep = 0; rep < repetitions; ++rep)
                {
                    size_t local_idx = 0;
                    process_block(block, local_idx);
                }
            }

            std::vector<StimLine> collect_block(const std::vector<StimLine> &lines,
                                                size_t &idx,
                                                int opening_line)
            {
                std::vector<StimLine> block;
                int depth = 1;
                while (idx < lines.size())
                {
                    const StimLine &candidate = lines[idx++];
                    std::string sanitized = trim(strip_comment(candidate.text));
                    if (sanitized.empty())
                    {
                        block.push_back(candidate);
                        continue;
                    }
                    if (sanitized == "}")
                    {
                        depth--;
                        if (depth == 0)
                        {
                            return block;
                        }
                        block.push_back(candidate);
                        continue;
                    }
                    if (sanitized.rfind("REPEAT", 0) == 0 && sanitized.find('{') != std::string::npos)
                    {
                        depth++;
                    }
                    block.push_back(candidate);
                }
                throw std::runtime_error("Stim parser error: unmatched '{' starting on line " + std::to_string(opening_line));
            }

            bool parse_index(const std::string &token, IdxType &value) const
            {
                if (token.empty())
                {
                    return false;
                }
                for (char ch : token)
                {
                    if (!std::isdigit(static_cast<unsigned char>(ch)))
                    {
                        return false;
                    }
                }
                value = static_cast<IdxType>(std::stoll(token));
                return true;
            }

            std::vector<IdxType> parse_targets(const StimLine &line,
                                               const std::vector<std::string> &tokens,
                                               size_t start_index) const
            {
                std::vector<IdxType> targets;
                for (size_t i = start_index; i < tokens.size(); ++i)
                {
                    IdxType value = -1;
                    if (!parse_index(tokens[i], value))
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): unsupported target '" + tokens[i] + "'");
                    }
                    targets.push_back(value);
                }
                return targets;
            }

            void update_max_qubit(const std::vector<IdxType> &targets)
            {
                for (auto target : targets)
                {
                    if (target < 0)
                    {
                        continue;
                    }
                    max_qubit_seen_ = std::max(max_qubit_seen_, target);
                }
                // Expand circuit immediately to avoid bounds check failures
                IdxType needed = max_qubit_seen_ + 1;
                if (needed > circuit_.num_qubits())
                {
                    circuit_.set_num_qubits(needed);
                }
            }

            void apply_multi_target_gate(const std::vector<IdxType> &targets,
                                         void (Circuit::*single)(IdxType),
                                         void (Circuit::*multi)(const std::vector<IdxType> &))
            {
                if (targets.empty())
                {
                    return;
                }
                if (targets.size() == 1)
                {
                    (circuit_.*single)(targets.front());
                    return;
                }
                if (multi != nullptr)
                {
                    (circuit_.*multi)(targets);
                }
                else
                {
                    for (auto target : targets)
                    {
                        (circuit_.*single)(target);
                    }
                }
            }

            void apply_single_gate_list(const std::vector<IdxType> &targets,
                                        void (Circuit::*fn)(IdxType))
            {
                for (auto target : targets)
                {
                    (circuit_.*fn)(target);
                }
            }

            void handle_instruction(const StimLine &line,
                                    const std::string &op_upper,
                                    const std::vector<std::string> &tokens)
            {
                if (op_upper == "TICK" ||
                    op_upper == "QUBIT_COORDS" ||
                    op_upper == "SHIFT_COORDS" ||
                    op_upper == "DETECTOR" ||
                    op_upper == "OBSERVABLE_INCLUDE" ||
                    op_upper == "REFLECTION" ||
                    op_upper == "END")
                {
                    return;
                }

                if (op_upper == "H")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    // Use single-qubit H gates (H_MULTI not supported by stabilizer backend)
                    for (auto target : targets)
                    {
                        circuit_.H(target);
                    }
                    return;
                }
                if (op_upper == "S")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    // Use single-qubit S gates (S_MULTI not supported by stabilizer backend)
                    for (auto target : targets)
                    {
                        circuit_.S(target);
                    }
                    return;
                }
                if (op_upper == "S_DAG" || op_upper == "S^-1" || op_upper == "S_INV")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    apply_single_gate_list(targets, &Circuit::SDG);
                    return;
                }
                if (op_upper == "CX" || op_upper == "CNOT")
                {
                    auto raw_targets = parse_targets(line, tokens, 1);
                    update_max_qubit(raw_targets);
                    if (raw_targets.size() < 2 || raw_targets.size() % 2 != 0)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): CX expects control-target pairs");
                    }
                    // Use individual CX gates (CX_MULTI not supported by stabilizer backend)
                    for (size_t i = 0; i + 1 < raw_targets.size(); i += 2)
                    {
                        circuit_.CX(raw_targets[i], raw_targets[i + 1]);
                    }
                    return;
                }
                if (op_upper == "CZ")
                {
                    auto raw_targets = parse_targets(line, tokens, 1);
                    update_max_qubit(raw_targets);
                    if (raw_targets.size() < 2 || raw_targets.size() % 2 != 0)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): CZ expects control-target pairs");
                    }
                    for (size_t i = 0; i + 1 < raw_targets.size(); i += 2)
                    {
                        circuit_.CZ(raw_targets[i], raw_targets[i + 1]);
                    }
                    return;
                }
                if (op_upper == "X")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    apply_single_gate_list(targets, &Circuit::X);
                    return;
                }
                if (op_upper == "Y")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    apply_single_gate_list(targets, &Circuit::Y);
                    return;
                }
                if (op_upper == "Z")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    apply_single_gate_list(targets, &Circuit::Z);
                    return;
                }
                if (op_upper == "R" || op_upper == "RESET")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    apply_single_gate_list(targets, &Circuit::RESET);
                    return;
                }
                if (op_upper == "M" || op_upper == "MR")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    const bool reset_after = (op_upper == "MR");
                    for (auto target : targets)
                    {
                        circuit_.M(target);
                        if (reset_after)
                        {
                            circuit_.RESET(target);
                        }
                    }
                    return;
                }

                // Handle parameterized instructions (noise, measurement with error, etc.)
                std::string base_op;
                std::vector<double> params;
                if (!parse_parameterized_op(tokens[0], base_op, params))
                {
                    throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): malformed instruction '" + tokens[0] + "'");
                }
                std::string base_op_upper = to_upper(base_op);

                // DEPOLARIZE1(p) - single-qubit depolarizing channel
                if (base_op_upper == "DEPOLARIZE1")
                {
                    if (params.size() != 1)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): DEPOLARIZE1 expects 1 parameter");
                    }
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    for (auto target : targets)
                    {
                        circuit_.DEP1(target, params[0]);
                    }
                    return;
                }

                // DEPOLARIZE2(p) - two-qubit depolarizing channel
                if (base_op_upper == "DEPOLARIZE2")
                {
                    if (params.size() != 1)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): DEPOLARIZE2 expects 1 parameter");
                    }
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    if (targets.size() < 2 || targets.size() % 2 != 0)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): DEPOLARIZE2 expects qubit pairs");
                    }
                    for (size_t i = 0; i + 1 < targets.size(); i += 2)
                    {
                        circuit_.DEP2(targets[i + 1], targets[i], params[0]);
                    }
                    return;
                }

                // PAULI_CHANNEL_1(px, py, pz) - custom single-qubit Pauli channel
                if (base_op_upper == "PAULI_CHANNEL_1")
                {
                    if (params.size() != 3)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): PAULI_CHANNEL_1 expects 3 parameters (px, py, pz)");
                    }
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    double px = params[0], py = params[1], pz = params[2];
                    double pi = 1.0 - px - py - pz;
                    std::vector<double> probs = {pi, px, py, pz};
                    for (auto target : targets)
                    {
                        circuit_.CHAN1(target, probs);
                    }
                    return;
                }

                // X_ERROR(p) - X error with probability p
                if (base_op_upper == "X_ERROR")
                {
                    if (params.size() != 1)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): X_ERROR expects 1 parameter");
                    }
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    double p = params[0];
                    std::vector<double> probs = {1.0 - p, p, 0.0, 0.0};
                    for (auto target : targets)
                    {
                        circuit_.CHAN1(target, probs);
                    }
                    return;
                }

                // Y_ERROR(p) - Y error with probability p
                if (base_op_upper == "Y_ERROR")
                {
                    if (params.size() != 1)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): Y_ERROR expects 1 parameter");
                    }
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    double p = params[0];
                    std::vector<double> probs = {1.0 - p, 0.0, p, 0.0};
                    for (auto target : targets)
                    {
                        circuit_.CHAN1(target, probs);
                    }
                    return;
                }

                // Z_ERROR(p) - Z error with probability p
                if (base_op_upper == "Z_ERROR")
                {
                    if (params.size() != 1)
                    {
                        throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): Z_ERROR expects 1 parameter");
                    }
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    double p = params[0];
                    std::vector<double> probs = {1.0 - p, 0.0, 0.0, p};
                    for (auto target : targets)
                    {
                        circuit_.CHAN1(target, probs);
                    }
                    return;
                }

                // MR(p) - Measure with error probability, then reset
                if (base_op_upper == "MR")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    for (auto target : targets)
                    {
                        circuit_.M(target);
                        // Apply X error after measurement if p > 0
                        if (!params.empty() && params[0] > 0)
                        {
                            std::vector<double> probs = {1.0 - params[0], params[0], 0.0, 0.0};
                            circuit_.CHAN1(target, probs);
                        }
                        circuit_.RESET(target);
                    }
                    return;
                }

                // M(p) - Measure with error probability
                if (base_op_upper == "M")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    for (auto target : targets)
                    {
                        circuit_.M(target);
                        // Apply X error after measurement if p > 0
                        if (!params.empty() && params[0] > 0)
                        {
                            std::vector<double> probs = {1.0 - params[0], params[0], 0.0, 0.0};
                            circuit_.CHAN1(target, probs);
                        }
                    }
                    return;
                }

                // MX(p) - X-basis measurement with error probability
                if (base_op_upper == "MX")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    for (auto target : targets)
                    {
                        circuit_.H(target);
                        circuit_.M(target);
                        if (!params.empty() && params[0] > 0)
                        {
                            std::vector<double> probs = {1.0 - params[0], params[0], 0.0, 0.0};
                            circuit_.CHAN1(target, probs);
                        }
                    }
                    return;
                }

                // RX - Reset to |+> state (R then H)
                if (base_op_upper == "RX")
                {
                    auto targets = parse_targets(line, tokens, 1);
                    update_max_qubit(targets);
                    for (auto target : targets)
                    {
                        circuit_.RESET(target);
                        circuit_.H(target);
                    }
                    return;
                }

                throw std::runtime_error("Stim parser error (line " + std::to_string(line.line_number) + "): unsupported instruction '" + tokens[0] + "'");
            }
        };
    } // namespace detail

    inline bool appendStimCircuitFromString(Circuit &circuit,
                                            const std::string &stim_source,
                                            int &n_qubits)
    {
        detail::StimParser parser(circuit, n_qubits);
        parser.parse_from_string(stim_source);
        n_qubits = static_cast<int>(parser.final_qubit_count());
        return true;
    }

    inline bool appendStimCircuitFromFile(Circuit &circuit,
                                          const std::string &filepath,
                                          int &n_qubits)
    {
        std::ifstream input(filepath);
        if (!input.is_open())
        {
            throw std::runtime_error("Unable to open Stim file: " + filepath);
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return appendStimCircuitFromString(circuit, buffer.str(), n_qubits);
    }
}
