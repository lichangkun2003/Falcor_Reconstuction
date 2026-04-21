import falcor

def render_graph_Pass():
    g = RenderGraph("VoxelReconstruction")

    voxelReconstruction_pass = createPass("VoxelReconstruction")
    voxel_pass = createPass("VoxelizationPass_CPU")
    read_pass = createPass("ReadVoxelPass")


    g.addPass(voxel_pass,"VoxelizationPass")
    g.addPass(read_pass,"ReadVoxelPass")
    g.addPass(voxelReconstruction_pass,"VoxelReconstruction")

    g.addEdge("VoxelizationPass.dummy","ReadVoxelPass.dummy")

    g.addEdge("ReadVoxelPass.vBuffer","VoxelReconstruction.vBuffer")
    #g.addEdge("ReadVoxelPass.gBuffer","VoxelReconstruction.gBuffer")
    #g.addEdge("ReadVoxelPass.pBuffer","VoxelReconstruction.pBuffer")
    g.addEdge("ReadVoxelPass.blockMap","VoxelReconstruction.blockMap")


    g.markOutput("VoxelReconstruction.dummy")

    return g

Graph = render_graph_Pass()
try: 
    m.addGraph(Graph)
except NameError: 
    pass
