import falcor

def render_graph_Pass():
    g = RenderGraph("VoxelReconstruction")

    voxelReconstruction_pass = createPass("VoxelReconstruction")


    g.addPass(voxelReconstruction_pass,"VoxelReconstruction")


    #g.addEdge("VoxelizationPass.dummy","ReadVoxelPass.dummy")


    #g.addEdge("RayMarchingPass.color","RenderToViewportPass.input")



    g.markOutput("VoxelReconstruction.dummy")

    return g

Graph = render_graph_Pass()
try: 
    m.addGraph(Graph)
except NameError: 
    pass
