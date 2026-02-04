import qldpc
import stim
from pathlib import Path


def generate_stim_circuit(distance: int, error_rate: float, num_rounds: int):
    d = distance
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",  # correct Z errors
        rounds=num_rounds,  # number of error correction rounds (typically d)
        distance=d,
    )
    noise_model = qldpc.circuits.SI1000NoiseModel(p=error_rate)
    return noise_model.noisy_circuit(circ)


def visualize_circuit(circ: stim.Circuit, output_dir: str = "."):
    """Visualize a Stim circuit in multiple formats"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Text diagram (console output)
    print("=== Text Timeline Diagram ===")
    print(circ.diagram("timeline-text"))
    print()

    # 2. Save SVG timeline diagram
    svg_timeline = str(circ.diagram("timeline-svg"))
    timeline_file = output_path / "circuit_timeline.svg"
    timeline_file.write_text(svg_timeline, encoding="utf-8")
    print(f"✓ Saved timeline SVG to: {timeline_file}")
    print("\n=== Circuit Statistics ===")
    print(f"Number of qubits: {circ.num_qubits}")
    print(f"Number of operations: {len(circ)}")
    print(f"Number of measurements: {circ.num_measurements}")
    print(f"Number of detectors: {circ.num_detectors}")
    print(f"Number of observables: {circ.num_observables}")


if __name__ == "__main__":
    # Generate a small circuit for visualization
    print("Generating surface code circuit...")
    circ = generate_stim_circuit(distance=3, error_rate=0.001, num_rounds=3)

    # Visualize it
    visualize_circuit(circ, output_dir="visualizations")

    print("\n💡 Open the SVG files in a web browser to view the diagrams!")
    print("   The 3D timeslice diagram is especially useful for surface codes.")
