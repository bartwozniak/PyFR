# -*- coding: utf-8 -*-

from collections import defaultdict, Counter
from itertools import chain, ifilter, izip
import re

import numpy as np

from pyfr.readers import BaseReader
from pyfr.readers.nodemaps import GmshNodeMaps
from pyfr.nputil import fuzzysort


def msh_section(mshit, section):
    endln = '$End{}\n'.format(section)
    endix = int(next(mshit)) - 1

    for i, l in enumerate(mshit):
        if l == endln:
            raise ValueError('Unexpected end of section $' + section)

        yield l

        if i == endix:
            break
    else:
        raise ValueError('Unexpected EOF')

    if next(mshit) != endln:
        raise ValueError('Expected $End' + section)


class GmshReader(BaseReader):
    # Supported file types and extensions
    name = 'gmsh'
    extn = ['.msh']

    # Gmsh element types to PyFR type (petype) and node counts
    _etype_map = {
        1: ('line', 2), 8: ('line', 3), 26: ('line', 4), 27: ('line', 5),
        2: ('tri', 3), 9: ('tri', 6), 21: ('tri', 10), 23: ('tri', 15),
        3: ('quad', 4), 10: ('quad', 9), 36: ('quad', 16), 37: ('quad', 25),
        4: ('tet', 4), 11: ('tet', 10), 29: ('tet', 20), 30: ('tet', 35),
        5: ('hex', 8), 12: ('hex', 27), 92: ('hex', 64), 93: ('hex', 125),
        6: ('pri', 6), 13: ('pri', 18), 90: ('pri', 40), 91: ('pri', 75),
        7: ('pyr', 5), 14: ('pyr', 14), 118: ('pyr', 30), 119: ('pyr', 55)
    }

    # Number of nodes in the first-order representation an element
    _petype_focount = {v[0]: v[1] for k, v in _etype_map.items() if k < 8}

    # Dimensionality of each element type
    _petype_ndim = {'tri': 2, 'quad': 2,
                    'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}

    # Face numberings for each element type
    _petype_fnums = {
        'tri': {'line': [0, 1, 2]},
        'quad': {'line': [0, 1, 2, 3]},
        'tet': {'tri': [0, 1, 2, 3]},
        'hex': {'quad': [0, 1, 2, 3, 4, 5]},
        'pri': {'quad': [2, 3, 4], 'tri': [0, 1]},
        'pyr': {'quad': [0], 'tri': [1, 2, 3, 4]}
    }

    # First-order node numbers associated with each element face
    _petype_fnmap = {
        'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
        'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
        'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
        'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                         [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
        'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
                'tri': [[0, 1, 2], [3, 4, 5]]}
    }

    def __init__(self, msh):
        if isinstance(msh, basestring):
            msh = open(msh)

        # Get an iterator over the lines of the mesh
        mshit = iter(msh)

        # Required section readers
        sect_map = {'MeshFormat': self._read_mesh_format,
                    'Nodes': self._read_nodes,
                    'Elements': self._read_eles,
                    'PhysicalNames': self._read_phys_names}
        req_sect = frozenset(sect_map)

        # Seen sections
        seen_sect = set()

        for l in ifilter(lambda l: l != '\n', mshit):
            # Ensure we have encountered a section
            if not l.startswith('$'):
                raise ValueError('Expected a mesh section')

            # Strip the '$' and '\n' to get the section name
            sect = l[1:-1]

            # If the section is known then read it
            if sect in sect_map:
                sect_map[sect](mshit)
                seen_sect.add(sect)
            # Else skip over it
            else:
                endsect = '$End{}\n'.format(sect)

                for el in mshit:
                    if el == endsect:
                        break
                else:
                    raise ValueError('Expected $End' + sect)

        # Check that all of the required sections are present
        if seen_sect != req_sect:
            missing = req_sect - seen_sect
            raise ValueError('Required sections: {} not found'
                             .format(missing))

    def _read_mesh_format(self, mshit):
        ver, ftype, dsize = next(mshit).split()

        if ver != '2.2':
            raise ValueError('Invalid mesh version')
        if ftype != '0':
            raise ValueError('Invalid file type')
        if dsize != '8':
            raise ValueError('Invalid data size')

        if next(mshit) != '$EndMeshFormat\n':
            raise ValueError('Expected $EndMeshFormat')

    def _read_phys_names(self, msh):
        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        #  - periodic faces
        self._felespent = None
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Extract the physical names
        for l in msh_section(msh, 'PhysicalNames'):
            m = re.match(r'(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError('Malformed physical entity')

            pent, name = int(m.group(2)), m.group(3).lower()

            # Fluid elements
            if name == 'fluid':
                self._felespent = pent
            # Periodic boundary faces
            elif name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p.group(1)].append(pent)
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

        if self._felespent is None:
            raise ValueError('No fluid elements in mesh')

        if any(len(pf) != 2 for pf in self._pfacespents.itervalues()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def _read_nodes(self, msh):
        self._nodepts = nodepts = {}

        for l in msh_section(msh, 'Nodes'):
            nv = l.split()
            nodepts[int(nv[0])] = np.array([float(x) for x in nv[1:]])

    def _read_eles(self, msh):
        elenodes = defaultdict(list)
        eleparts = defaultdict(list)

        for l in msh_section(msh, 'Elements'):
            # Extract the raw element data
            elei = [int(i) for i in l.split()]
            enum, etype, entags = elei[:3]
            etags, enodes = elei[3:3 + entags], elei[3 + entags:]

            if etype not in self._etype_map:
                raise ValueError('Unsupported element type {}'.format(etype))

            # Physical entity type (used for BCs)
            epent = etags[0]

            # Determine the partition number (defaults to 0)
            epart = etags[3] - 1 if entags > 2 else 0

            elenodes[etype, epent].append(enodes)
            eleparts[etype, epent].append(epart)

        self._elenodes = {k: np.array(v) for k, v in elenodes.iteritems()}
        self._eleparts = {k: np.array(v) for k, v in eleparts.iteritems()}

    def _to_first_order(self, elemap):
        foelemap = {}
        for (etype, epent), eles in elemap.iteritems():
            # PyFR element type ('hex', 'tri', &c)
            petype = self._etype_map[etype][0]

            # Number of nodes in the first-order representation
            focount = self._petype_focount[petype]

            foelemap[petype, epent] = eles[:,:focount]

        return foelemap

    def _split_fluid(self, elemap):
        selemap = defaultdict(dict)

        for (petype, epent), eles in elemap.iteritems():
            selemap[epent][petype] = eles

        return selemap.pop(self._felespent), selemap

    def _foface_array(self, petype, pftype, foeles):
        # Face numbers of faces of this type on this element
        fnums = self._petype_fnums[petype][pftype]

        # First-order nodes associated with this face
        fnmap = self._petype_fnmap[petype][pftype]

        # Number of first order nodes needed to define a face of this type
        nfnodes = self._petype_focount[pftype]

        dtype = [('petype', 'S4'), ('eidx', 'i4'), ('fidx', 'i1'),
                 ('flags', 'i1'), ('nodes', 'i8', nfnodes)]

        arr = np.recarray((len(foeles), len(fnums)), dtype=dtype)
        arr.petype = petype
        arr.eidx = np.arange(len(foeles))[...,None]
        arr.fidx = fnums
        arr.nodes = foeles[:,fnmap]
        arr.flags = 0

        return arr.ravel()

    def _extract_faces(self, foeles):
        fofaces = defaultdict(list)
        for petype, eles in foeles.iteritems():
            for pftype in self._petype_fnums[petype]:
                fofarr = self._foface_array(petype, pftype, eles)
                fofaces[pftype].append(fofarr)

        return fofaces

    def _pair_fluid_faces(self, ffofaces):
        pairs = defaultdict(list)
        resid = {}

        for pftype, faces in ffofaces.iteritems():
            for f in chain(*faces):
                sn = tuple(sorted(f['nodes']))

                # See if the nodes are in resid
                if sn in resid:
                    pairs[pftype].append((resid.pop(sn), f))
                # Otherwise add them to the unpaired dict
                else:
                    resid[sn] = f

        return pairs, resid

    def _pair_periodic_fluid_faces(self, bpart, resid):
        pfaces = defaultdict(list)
        nodepts = self._nodepts

        for lpent, rpent in self._pfacespents.itervalues():
            for pftype in bpart[lpent]:
                lfnodes = bpart[lpent][pftype]
                rfnodes = bpart[rpent][pftype]

                lfpts = np.array([[nodepts[n] for n in fn] for fn in lfnodes])
                rfpts = np.array([[nodepts[n] for n in fn] for fn in rfnodes])

                lfidx = fuzzysort(lfpts.mean(axis=1).T, xrange(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, xrange(len(rfnodes)))

                for lfn, rfn in izip(lfnodes[lfidx], rfnodes[rfidx]):
                    lf = resid.pop(tuple(sorted(lfn)))
                    rf = resid.pop(tuple(sorted(rfn)))

                    pfaces[pftype].append((lf, rf))

        return pfaces

    def _ident_boundary_faces(self, bpart, resid):
        bfaces = defaultdict(list)

        bpents = set(self._bfacespents.itervalues())

        for epent, fnodes in bpart.iteritems():
            if epent in bpents:
                for fn in chain.from_iterable(fnodes.itervalues()):
                    bfaces[epent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def _partition_pairs(self, pairs, bcf):
        con_px = defaultdict(list)
        con_pxpy = defaultdict(list)
        bcon_px = defaultdict(list)

        # Connectivity in PyFR is specified in terms of partition-local
        # element numbers.  As the element indices in pairs are
        # global it is first necessary to produce a global-to-local
        # mapping for each element type.
        eleglmap = defaultdict(list)
        pcounter = Counter()

        feleparts = self._split_fluid(self._eleparts)[0]
        for etype, eleps in feleparts.iteritems():
            petype = self._etype_map[etype][0]

            for p in eleps:
                eleglmap[petype].append((p, pcounter[petype, p]))
                pcounter[petype, p] += 1

        # Generate the face connectivity
        for l, r in pairs:
            lpetype, leidxg, lfidx, lflags, lnodes = l
            rpetype, reidxg, rfidx, rflags, rnodes = r

            lpart, leidxl = eleglmap[lpetype][leidxg]
            rpart, reidxl = eleglmap[rpetype][reidxg]

            conl = (lpetype, leidxl, lfidx, lflags)
            conr = (rpetype, reidxl, rfidx, rflags)

            if lpart == rpart:
                con_px[lpart].append([conl, conr])
            else:
                con_pxpy[lpart, rpart].append(conl)
                con_pxpy[rpart, lpart].append(conr)

        # Generate boundary conditions
        for pbcrgn, pent in self._bfacespents.iteritems():
            for lpetype, leidxg, lfidx, lflags, lnodes in bcf[pent]:
                lpart, leidxl = eleglmap[lpetype][leidxg]
                conl = (lpetype, leidxl, lfidx, 0)

                bcon_px[pbcrgn, lpart].append(conl)

        return con_px, con_pxpy, bcon_px

    def _get_connectivity(self):
        # For connectivity a first-order representation is sufficient
        eles = self._to_first_order(self._elenodes)

        # Split into fluid and boundary parts
        fpart, bpart = self._split_fluid(eles)

        # Extract the faces of the fluid elements
        ffaces = self._extract_faces(fpart)

        # Pair the fluid-fluid faces
        fpairs, resid = self._pair_fluid_faces(ffaces)

        # Identify periodic boundary face pairs
        pfpairs = self._pair_periodic_fluid_faces(bpart, resid)

        # Identify the fixed boundary faces
        bf = self._ident_boundary_faces(bpart, resid)

        if any(resid.itervalues()):
            raise ValueError('Unpaired faces in mesh')

        # Flattern the face-pair dicts
        pairs = chain(chain.from_iterable(fpairs.itervalues()),
                      chain.from_iterable(pfpairs.itervalues()))

        # Process these face pairs into the connectivity arrays
        con_px, con_pxpy, bcon_px = self._partition_pairs(pairs, bf)

        # Output
        retcon = {}

        for k, v in con_px.iteritems():
            retcon['con_p%d' % k] = np.array(v, dtype='S5,i4,i1,i1').T

        for k, v in con_pxpy.iteritems():
            retcon['con_p%dp%d' % k] = np.array(v, dtype='S5,i4,i1,i1')

        for k, v in bcon_px.iteritems():
            retcon['bcon_%s_p%d' % k] = np.array(v, dtype='S5,i4,i1,i1')

        return retcon

    def _get_shape_points(self):
        spts = defaultdict(list)

        # Global node map (node index to coords)
        nodepts = self._nodepts

        for etype, pent in self._elenodes:
            if pent != self._felespent:
                continue

            # Elements and corresponding partition numbers
            eles = self._elenodes[etype, pent]
            prts = self._eleparts[etype, pent]

            petype, nnodes = self._etype_map[etype]

            # Go from Gmsh to PyFR node ordering
            peles = eles[:,GmshNodeMaps.from_pyfr[petype, nnodes]]

            # Obtain the dimensionality of the element type
            ndim = self._petype_ndim[petype]

            for nn, p in izip(peles, prts):
                spts[petype, p].append([nodepts[i][:ndim] for i in nn])

        return {'spt_{}_p{}'.format(*k): np.array(arr).swapaxes(0, 1)
                for k, arr in spts.iteritems()}

    def _to_raw_pyfrm(self):
        rawm = {}
        rawm.update(self._get_connectivity())
        rawm.update(self._get_shape_points())
        return rawm
