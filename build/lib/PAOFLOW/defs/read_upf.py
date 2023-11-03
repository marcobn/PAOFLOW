#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO

# read UPF utility from Davide Ceresoli
class UPF:

  def __init__ ( self, filename ):
    """Open a UPF file, determine version, and read it"""
    with open(filename) as f:
      xml_file_content = f.read()

    # fix broken XML
    xml_file_content = xml_file_content.replace('&', '&amp;')

    # test if v1
    if xml_file_content.startswith('<PP_INFO>'):
      xml_file_content = '<UPF version="1.0">\n' + xml_file_content + '</UPF>\n'

    # parse the XML file
    root = ET.fromstring(xml_file_content)

    # dispatch to the specific routine
    upfver = root.attrib["version"]
    self.version = int(upfver.split(".")[0])
    if self.version == 1:
      self._read_upf_v1(root)
    elif self.version == 2:
      self._read_upf_v2(root)
    else:
      raise RuntimeError('Wrong UPF version: %s'%upfver)


  def _read_upf_v1 ( self, root ):
    """Read a UPF v1 pseudopotential"""

    # parse info and header
    self.info = root.find('PP_INFO').text
    for line in root.find('PP_HEADER').text.split('\n'):
      l = line.split()
      if 'Element' in line:       self.element = l[0]
      if 'NC' in line:            self.ptype = 'NC'
      if 'US' in line:            self.ptype = 'US'
      if 'Nonlinear' in line:     self.nlcc = l[0] == 'T'
      if 'Exchange' in line:      self.qexc = ' '.join(l[0:4])
      if 'Z valence' in line:     self.val = float(l[0])
      if 'Max angular' in line:   self.lmax = int(l[0])
      if 'Number of po' in line:  self.npoints = int(l[0])
      if 'Number of Wave' in line:
        self.nwfc = int(l[0])
        self.nproj = int(l[1])

    # parse mesh
    text = root.find('PP_MESH/PP_R').text
    self.r = np.array( [float(x) for x in text.split()] )
    text = root.find('PP_MESH/PP_RAB').text
    self.rab = np.array( [float(x) for x in text.split()] )

    # local potential
    text = root.find('PP_LOCAL').text
    self.vloc = np.array( [float(x) for x in text.split()] ) / 2.0  # to Hartree

    # atomic wavefunctions
    self.pswfc = []
    chis = root.find('PP_PSWFC')
    self.shells = []
    self.jchia = []
    if chis is not None:
      data = StringIO(chis.text)
      nlines = self.npoints//4
      if self.npoints % 4 != 0: nlines += 1

      while True:
        line = data.readline()
        if line == '\n': continue
        if line == '': break
        label, l, occ, dummy = line.split()
 
        wfc = []
        for i in range(nlines):
          wfc.extend(map(float, data.readline().split()))
        wfc = np.array(wfc)
        self.shells.append(int(l))
        self.pswfc.append( {'label': label, 'occ': float(occ), 'wfc': wfc} )

    # atomic rho
    self.atrho = None
    atrho = root.find('PP_RHOATOM')
    if atrho is not None:
      self.atrho = np.array( [float(x) for x in atrho.text.split()] )

    # TODO: NLCC

    # TODO: PS_NONLOCAL/BETA, PP_DIJ

    # TODO: GIPAW data


  def _read_upf_v2 ( self, root ):
    """Read a UPF v2 pseudopotential"""

    # parse header
    h = root.find('PP_HEADER').attrib
    self.element = h['element']
    self.type = h['pseudo_type']
    self.nlcc = h['core_correction'] == 'true'
    self.qexc = h['functional']
    self.val = float(h['z_valence'])
    self.lmax = int(h['l_max'])
    self.npoints = int(h['mesh_size'])
    self.nwfc = int(h['number_of_wfc'])
    self.nproj = int(h['number_of_proj'])
    self.v2_header = h.copy()

    # parse mesh
    text = root.find('PP_MESH/PP_R').text
    self.r = np.array( [float(x) for x in text.split()] )
    text = root.find('PP_MESH/PP_RAB').text
    self.rab = np.array( [float(x) for x in text.split()] )

    # local potential
    text = root.find('PP_LOCAL').text
    self.vloc = np.array( [float(x) for x in text.split()] ) / 2.0  # to Hartree

    # atomic wavefunctions
    self.pswfc = []
    self.jchia = []
    self.shells = []
    i = 0
    while True:
      i += 1
      chi = root.find('PP_PSWFC/PP_CHI.%i' % (i))
      if chi is None: break

      label = chi.attrib["label"]
      occ = float(chi.attrib["occupation"])
      wfc = [float(x) for x in chi.text.split()]
      wfc = np.array(wfc)
      self.shells.append(int(chi.attrib['l']))
      self.pswfc.append( {'label': label, 'occ': float(occ), 'wfc': wfc} )

      jchi = root.find('PP_SPIN_ORB/PP_RELWFC.%d'%i)
      if jchi is not None:
        self.jchia.append(float(jchi.attrib['jchi']))
    self.jchia = self.jchia
    self.shells = self.shells

    # TODO: NLCC, ATRHO

    # TODO: PS_NONLOCAL/BETA, PP_DIJ

    # TODO: GIPAW data
