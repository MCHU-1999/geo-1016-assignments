import sys
import time
import polyfit

# Define the file paths
POLYFIT_FOLDER = "./A3_Reconstruction_Data/surface_reconstruction/polygonal_surface_reconstruction"
POLYFIT_FILES = [
    "PolyFit_01_polyhedron.bvg",
    "PolyFit_02_SyntheticBuilding.bvg",
    "PolyFit_03_LongBuilding(Fitting0.4-Coverage0.3.-Complexity0.3).bvg",
    "PolyFit_04_foampack.bvg"
]

if __name__ == "__main__":
    # Initialize PolyFit
    polyfit.initialize()

    for file_name in POLYFIT_FILES:
        input_file = f"{POLYFIT_FOLDER}/{file_name}"
        
        print(f"Processing: {input_file}")
        
        # Load the point cloud
        point_cloud = polyfit.read_point_set(input_file)
        if not point_cloud:
            print(f"Failed to load point cloud from {input_file}", file=sys.stderr)
            continue
        
        # Start timing
        start_time = time.time()
        
        # Perform the reconstruction
        mesh = polyfit.reconstruct(
            point_cloud,  # Input point cloud
            polyfit.SCIP, # Solver (use GUROBI if licensed)
            0.43,  # Weight of data fitting
            0.27,  # Weight of model coverage
            0.3    # Weight of model complexity
        )
        
        if not mesh:
            print(f"Reconstruction failed for {input_file}", file=sys.stderr)
            continue
        
        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Reconstruction completed in {elapsed_time:.2f} seconds.")
        print(f"Reconstructed mesh has {mesh.size_of_facets()} faces.")
