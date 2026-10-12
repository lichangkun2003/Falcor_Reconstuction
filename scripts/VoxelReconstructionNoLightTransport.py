import falcor

def render_graph_Pass():
    g = RenderGraph("VoxelReconstructionNoLightTransport")

    VoxelReconstructionNoLightTransport_pass = createPass("VoxelReconstructionNoLightTransport")
    accumulate_pass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single",'maxFrameCount': 1024})
    g.addPass(VoxelReconstructionNoLightTransport_pass,"VoxelReconstruction")
    g.addPass(accumulate_pass,"AccumulatePass")

    g.addEdge("VoxelReconstruction.color","AccumulatePass.input")
    g.markOutput("VoxelReconstruction.color")
    g.markOutput("VoxelReconstruction.AccuColor")
    g.markOutput("AccumulatePass.output")
    g.markOutput("VoxelReconstruction.dummy")
    return g

Graph = render_graph_Pass()
try:
    m.addGraph(Graph)
except NameError:
    pass
