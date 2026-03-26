import cProfile
import block_blast_simulator

if __name__ == "__main__":
    cProfile.run("block_blast_simulator.run_benchmark()", sort="cumtime")
