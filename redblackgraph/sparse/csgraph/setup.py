def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('csgraph', parent_package, top_path)

    config.add_extension('_shortest_path',
                         sources=['_shortest_path.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('_rbg_math',
                         sources=['_rbg_math.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('_components',
                         sources=['_components.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('_permutation',
                         sources=['_permutation.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('_ordering',
                         sources=['_ordering.c'],
                         include_dirs=[numpy.get_include()])

    # config.add_extension('_traversal',
    #                      sources=['_traversal.c'],
    #                      include_dirs=[numpy.get_include()])
    #
    # config.add_extension('_min_spanning_tree',
    #                      sources=['_min_spanning_tree.c'],
    #                      include_dirs=[numpy.get_include()])
    #
    # config.add_extension('_matching',
    #                      sources=['_matching.c'],
    #                      include_dirs=[numpy.get_include()])
    #
    # config.add_extension('_flow',
    #                      sources=['_flow.c'],
    #                      include_dirs=[numpy.get_include()])
    #
    # config.add_extension('_reordering',
    #                      sources=['_reordering.c'],
    #                      include_dirs=[numpy.get_include()])
    #
    config.add_extension('_tools',
                         sources=['_tools.c'],
                         include_dirs=[numpy.get_include()])

    return config
