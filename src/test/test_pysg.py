#! /usr/bin/env python
#  Modified:
#
#    23 February 2016
#
#  Author:
#
#    Jochen Garcke
#
#  Reference:
#
#    Jochen Garcke,
#    A sparse grid tutorial.
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import math  # noqa:F401

import src.pysg as pysg


def test_SGNoBound3D():
    #  SG is a sparse grid of dimension 3 and level 3.
    #  Create sg.indices which stores level and position for each point.
    sg = pysg.sparseGrid(3, 3)

    #  Determine sg.gP with the coordinates of the points
    #  associated with the sparse grid index set.
    sg.generatePoints()

    #  Print the points in the grid.
    print("Coordinates of points in 3D sparse grid of level 3.")

    for i in range(len(sg.indices)):
        sg.gP[tuple(sg.indices[i])].printPoint()

    #  Did we compute the right number of grid points?
    assert len(sg.indices) == 31

    #  Evaluate 4x(1-x)*4y(1-y)*4z(1-z) at each grid point.
    for i in range(len(sg.indices)):
        sum = 1.0
        pos = sg.gP[tuple(sg.indices[i])].pos
        for j in range(len(pos)):
            sum *= 4.0 * pos[j] * (1.0 - pos[j])
        sg.gP[tuple(sg.indices[i])].fv = sum

    # #  Convert the sparse grid from nodal to hierarchical values.
    # sg.nodal2Hier()
    #
    # #  Does the evaluation of the sparse grid function in
    # #  hierarchical values give the correct value gv?
    # for i in range(len(sg.indices)):
    #     assert sg.gP[tuple(sg.indices[i])].fv == sg.evalFunct(
    #         sg.gP[tuple(sg.indices[i])].pos
    #     )


def test_SGNoBound2D():
    #  SG is a sparse grid of dimension 2 and level 3.
    #  Create sg.indices which stores level and position for each point.
    sg = pysg.sparseGrid(2, 3)

    #  Determine sg.gP with the coordinates of the points
    #  associated with the sparse grid index set.
    sg.generatePoints()

    #  Print the points in the grid.
    print("Coordinates of points in 2D sparse grid of level 3.")

    for i in range(len(sg.indices)):
        sg.gP[tuple(sg.indices[i])].printPoint()

    #  Did we compute the right number of grid points?
    assert len(sg.indices) == 17

    #  Evaluate 4x(1-x)*4y(1-y) at each grid point.
    for i in range(len(sg.indices)):
        sum = 1.0
        pos = sg.gP[tuple(sg.indices[i])].pos
        for j in range(len(pos)):
            sum *= 4.0 * pos[j] * (1.0 - pos[j])
        sg.gP[tuple(sg.indices[i])].fv = sum

    # #  Convert to hierarchical values.
    # sg.nodal2Hier()
    #
    # #  Does the evaluation of sparse grid function in
    # #  hierarchical values give the correct value gv?
    # for i in range(len(sg.indices)):
    #     assert sg.gP[tuple(sg.indices[i])].fv == sg.evalFunct(
    #         sg.gP[tuple(sg.indices[i])].pos
    #     )
