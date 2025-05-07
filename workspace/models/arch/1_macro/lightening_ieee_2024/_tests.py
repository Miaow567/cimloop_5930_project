import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on

def test_explore_architectures(model_name: str, prf: int):
    model_dir = utl.path_from_model_dir(f"workloads/{model_name}")
    layer_paths = [
        os.path.join(model_dir, l) for l in os.listdir(model_dir) if l.endswith(".yaml")
    ]
    layer_paths = [l for l in layer_paths if "From einsum" not in open(l, "r").read()]

    def callfunc(spec):
        # Reduce compile time by relaxing convergence
        spec.mapper.victory_condition = 10

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            variables=dict(
                N_TILES=n_tiles,
                N_DPTC_PER_TILE=n_dptc,
                N_DPTC=n_tiles * n_dptc,
                ENCODED_INPUT_BITS=input_bits,
                ENCODED_WEIGHT_BITS=weight_bits,
                ENCODED_OUTPUT_BITS=8,
                TEMPORAL_DAC_RESOLUTION=temporal_dac,
                VOLTAGE_DAC_RESOLUTION=voltage_dac,
                DAC_RESOLUTION=max(voltage_dac, temporal_dac),
                BASE_LATENCY=2.0e-10,
                LATENCY_SCALING_SCALE=0.625,
                BITS_PER_CELL=4,
                CIM_ARCHITECTURE=True,
                SIGNED_SUM_ACROSS_INPUTS=True,
                SIGNED_SUM_ACROSS_WEIGHTS=True,
                INPUT_ENCODING_FUNC="offset_encode_hist",
                WEIGHT_ENCODING_FUNC="two_part_magnitude_encode_if_signed_hist",
            ),
            system="ws_dummy_buffer_one_macro",
            callfunc=callfunc,
        )
        for l in layer_paths[:prf]
        for n_tiles, n_dptc in [(2, 2), (4, 4)]
        for input_bits in [4, 8]
        for weight_bits in [4, 8]
        for temporal_dac, voltage_dac in [(1, 8), (4, 4)]
    )

    results.combine_per_component_energy(
        ["input_dac", "input_modulator"], "Input Conversion"
    )
    results.combine_per_component_energy(
        ["weight_dac", "weight_modulator"], "Weight Conversion"
    )
    results.combine_per_component_energy(
        ["photodetector_array", "accumulation_unit", "output_adc"], "Output Conversion"
    )
    results.combine_per_component_energy(
        ["laser_source", "micro_comb", "optical_awg"], "Optical Routing"
    )
    results.combine_per_component_energy(["global_buffer"], "Global Memory")
    results.clear_zero_energies()

    return results


def test_transformer(dnn_name: str, prf: int):
    dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]

    def callfunc(spec):
        spec.mapper.victory_condition = 10

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=layer_path,
            variables=dict(
                BATCH_SIZE=batch_size,
            ),
            system="ws_dummy_buffer_one_macro",
            callfunc=callfunc,
        )
        for batch_size in [1, 8]
        for layer_path in layer_paths[:prf]
    )
    return results

if __name__ == "__main__":
    test_energy_breakdown()
    test_area_breakdown()
    test_full_dnn("alexnet")
    test_full_dnn("vgg16")
    test_explore_architectures("resnet18")
    test_explore_main_memory("resnet18")