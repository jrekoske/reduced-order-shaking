#!/usr/bin/env python3

import numpy as np

import trimesh
import seissolxdmf as sx
import seissolxdmfwriter as sw

# These 2 latter modules are on pypi (e.g. pip install seissolxdmf)


def read_reshape2d(sx, dataname):
    """read seissol dataset
    and if there is only one time stamp
    create a second dimension of size 1"""
    myData = sx.ReadData(dataname)
    if len(myData.shape) == 1:
        myData = myData.reshape((1, myData.shape[0]))
    return myData


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    """
    return indexes of sorted points robust to small perturbations of individual
    components.
    https://stackoverflow.com/questions/19072110/
    numpy-np-lexsort-with-fuzzy-tolerant-comparisons
    note that I added dim<arr.shape[0]-1 in some if statement
    (else it will crash sometimes)
    """
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    i, ix = 0, srtdidx[0]
    for j, jx in enumerate(srtdidx[1:], start=1):
        if arrd[jx] - arrd[ix] >= tol:
            if j - i > 1 and dim < arr.shape[0] - 1:
                srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
            i, ix = j, jx

    if i != j and dim < arr.shape[0] - 1:
        srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


def lookup_sorted_geom(geom):
    """return the indices to sort the
    geometry array by x, then y, then z
    and the associated inverse look-up table
    """
    ind = fuzzysort(geom.T, list(range(0, geom.shape[0])), tol=1e-4)
    # generate inverse look-up table
    dic = {i: index for i, index in enumerate(ind)}
    ind_inv = np.zeros_like(ind)
    for k, v in dic.items():
        ind_inv[v] = k
    return ind, ind_inv


def read_geom_connect(sx):
    return sx.ReadGeometry(), sx.ReadConnect()


def return_sorted_geom_connect(sx):
    """sort geom array and reindex connect array to match the new geom array"""
    geom, connect = read_geom_connect(sx)
    nv = geom.shape[0]
    trimesh.tol.merge = 1e-4
    mesh = trimesh.Trimesh(geom, connect)
    mesh.merge_vertices()
    geom = mesh.vertices
    connect = mesh.faces
    print(f"removed {nv-geom.shape[0]} duplicates out of {nv}")

    ind, ind_inv = lookup_sorted_geom(geom)
    geom = geom[ind, :]
    connect = np.array([ind_inv[x]
                       for x in connect.flatten()]).reshape(connect.shape)
    # sort along line (then we can use multidim_intersect)
    connect = np.sort(connect, axis=1)
    return geom, connect


def multidim_intersect(arr1, arr2):
    """find indexes of same triangles in 2 connect arrays
    (associated with the same geom array)
    generate 1D arrays of tuples and use numpy function
    https://stackoverflow.com/questions/9269681/
    intersection-of-2d-numpy-ndarrays
    """
    arr1_view = arr1.view([("", arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([("", arr2.dtype)] * arr2.shape[1])
    intersected, ind1, ind2 = np.intersect1d(
        arr1_view, arr2_view, return_indices=True)
    ni, n1, n2 = intersected.shape[0], arr1.shape[0], arr2.shape[0]
    print(
        f"{ni} faces in common, n faces connect 1:{n1}, 2:{n2}"
        " (diff: {n1-ni}, {n2-ni})"
    )
    return ind1, ind2


def run_reordering(xdmf_filename1, xdmf_filename2, idt, data, ratio=False):
    sx1 = sx.seissolxdmf(xdmf_filename1)
    sx2 = sx.seissolxdmf(xdmf_filename2)

    geom1, connect1 = return_sorted_geom_connect(sx1)
    geom2, connect2 = return_sorted_geom_connect(sx2)
    if not np.all(np.isclose(geom1, geom2, rtol=1e-3, atol=1e-4)):
        raise ValueError("geometry arrays differ")
    ind1, ind2 = multidim_intersect(connect1, connect2)
    connect1 = connect1[ind1, :]

    print('idt:', idt)
    if idt[0] == -1:
        idt = list(range(0, sx1.ndt))
    print('idt:', idt)

    aData = []
    if data == ["all"]:
        variable_names = set()
        for elem in sx1.tree.iter():
            if elem.tag == "Attribute":
                variable_names.add(elem.get("Name"))
        variable_names2 = set()
        for elem in sx2.tree.iter():
            if elem.tag == "Attribute":
                variable_names2.add(elem.get("Name"))
        # return only variables in common
        variable_names = variable_names.intersection(variable_names2)
        for to_remove in ["partition", "locationFlag"]:
            if to_remove in variable_names:
                variable_names.remove(to_remove)
    else:
        variable_names = data

    for dataname in variable_names:
        print(dataname)
        myData1 = read_reshape2d(sx1, dataname)
        myData2 = read_reshape2d(sx2, dataname)
        ndt = min(myData1.shape[0], myData2.shape[0])
        myData = myData2[0:ndt, ind2]

        print('idt:', idt)
        for idt_val in idt:
            if idt_val < ndt:
                pass
            else:
                idt.pop(idt_val)

        aData.append(myData)

    fname = 'loh1-GME_corrected'

    try:
        dt = sx1.ReadTimeStep()
    except NameError:
        dt = 0.0
    sw.write_seissol_output(
        fname, geom1, connect1, variable_names, aData, dt, idt)
