import gffx

def test_graph_conv():
    graph_conv_layer = gffx.GraphConv()

    N = 1024
    A = gffx.random.adjmat(N, 6)

    breakpoint()