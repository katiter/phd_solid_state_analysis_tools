#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created 2014 by Kathy Chen

Investigate the band structure of hybrid systems by partitioning the system into its inorganic and organic halves, then:
- Get the energies of the frontier orbitals
- Get the Fermi level and realign it to the vacuum level
- Plot the 4-D structure


Note that dos._site_dos returns all orbitals, whereas dos.site_dos gets a specific orbital set dos.site_dos automatically adds one to the orbital index. In your case, it is usually n==10, therefore:

>>> norb = {'s':1, 'py':2, 'pz':3, 'px':4, 'dxy':5, 'dyz':6, 'dz2':7, 'dxz':8, 'dx2':9}

# Load some sample calculations
>>> calcpath = '~Project/espresso_adsys/ana_dteccdte_bz_oo'
>>> dos = ldos.LdosInfo(calcpath);
>>> dos.plot()
>>> dos.zoom_to_molecule()
>>> dos.zoom(xmin = -10, xmax = 4.5, ymax = 30)
>>> dos.show_homolumo()
>>> dos.write_name()
>>> print "Saving to " + plotpath
>>> dos.save(plotpath)
>>> dos.annotate_lumos(lookahead=2, delta=0.1)
>>> dos.draw()

To get a specific part of the DOS to play with:
>>> oxygen_s = dos.pdos_collection.get(PdosParts.Surface_Oxygens).s

"""

import sys
import os
from os.path import join
import numpy
import ase.calculators.vasp as vcalc
from numpy import array
from optparse import OptionParser
import matplotlib.pyplot as plt
import warnings
import asethings


from collections import namedtuple
from enum import Enum
Pdos = namedtuple('Pdos', 'tot s p d')
Vaclevel = namedtuple('Vaclevel', 'lower upper')
OrbitalInfo = namedtuple('OrbitalInfo', 'energy index height')
#This is the string format used in the print_info() command.
INFO_FORMAT = "{:<40}{:<10}{:<20}{:<10}{:<35}{:>15}\n"
TESTDIR = join(os.path.dirname(os.path.realpath(sys.argv[0])), "test")


class PdosParts(Enum):
  """
  These are the only things that can go into 'self.pdos_collection'
  They're numbered so that when plotted, their z-order is already known.
  """
  Value = namedtuple('Value', 'zOrder, color')
  Surface = Value(0, "0xDCDCDC")
  Surface_Oxygens = Value(1, "0xBABABA")
  #These oxygens are at least 6 angstroms away from the molecule anchor
  Surface_Oxygens_Not_Close = Value(2, "0xA9A9A9") 
  molecule = Value(3, "0xFF8C00")
  #This is the molecule excluding the anchor
  Photocore = Value(4, "0x6699FF") 
  Anchor = Value(5, "0x8B0000")


def read_fermi(outcar_file):
  """NOT USED: Method that reads the last occurance of Fermi energy from OUTCAR file"""
  E_f = None
  for line in open(outcar_file, 'r'):
      if line.rfind('E-fermi') > -1:
          E_f=float(line.split()[2])
  return E_f

def read_nelect(outcar_file):
  """Method that reads the number of electrons. Note that alternatively, If you look at dos_data.integrated_dos, you can see that at the Fermi energy it gives you the total number of electrons.
"""
  E_f = None
  for line in open(outcar_file, 'r'):
      if line.rfind('NELECT') > -1:
          E_f=float(line.split()[2])
  return E_f

def get_total_dos(dos_data):
  dos_total = dos_data.dos
  return dos_total

def get_dos_of_these_atoms(indices, dos_data):
  if indices == None or len(indices) == 0:
    return None, None, None, None
  num_orbitals = dos_data._site_dos.shape[1] - 1
  dos_tot = dos_s = dos_p = dos_d = numpy.zeros(dos_data.dos.shape)
  for atom in indices:
    dos_tot = dos_tot + sum([dos_data.site_dos(atom, i) for i in range(0,num_orbitals)])
    dos_s = dos_s + sum([dos_data.site_dos(atom, i) for i in range(0, 1)])
    dos_p = dos_p + sum([dos_data.site_dos(atom, i) for i in range(1, 4)])
    dos_d = dos_d + sum([dos_data.site_dos(atom, i) for i in range(4, 9)])
  return Pdos(dos_tot, dos_s, dos_p, dos_d)


def get_special_dos_of_these_atoms(indices, dos_data, orbitals=None):
  """ If the phase factors have been calculated, then it's like this:
  ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2']
  """
  all_orbitals = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2']
  if orbitals==None:
    print('Select some orbitals from ', all_orbitals)
    return
  dos_result = numpy.zeros(dos_data.dos.shape)
  orbital_indices = map(lambda orb: all_orbitals.index(orb), orbitals)
  for atom in indices:
    dos_result = dos_result + sum([dos_data.site_dos(atom, i) for i in orbital_indices])
  return dos_result


def peak_detect(x_energies, y_dos, lookahead=2, delta=0):
  from peakdetect import peakdetect
  peaks, valleys = map(array, peakdetect(y_dos, x_energies, lookahead=lookahead, delta=delta))
  return peaks

def get_vaclevel_from_outcar(outcar_file):
  with open(outcar_file, 'r') as f:
    for line in f:
      if 'vacuum' in line:
        vac_upper, vac_lower = map(float, line.split()[-2:])
        return Vaclevel(vac_lower, vac_upper)

def _read_locpot(locpot_file):
  """ If you run chg1d.py LOCPOT z, it produces a LOCPOT_Z file. 
  This method takes in the LOCPOT_Z file and gives the zipped arrays corresponding to z-height and potential energy
  """
  locpot = []
  LocpotEntry = namedtuple('LocpotEntry', 'zheight potential')
  with open(locpot_file, 'r') as f:
    f.readline() #Get past the header
    for line in f:
      locpot.append(LocpotEntry(*map(float, line.split())))
  return locpot

def get_vaclevel_from_locpot(locpot_file, plot=False, verbose=True,
  within_angstroms=2, drop_angstroms_from_edges=0.2):
  locpot = _read_locpot(locpot_file)
  z_lowest = locpot[0].zheight + drop_angstroms_from_edges
  z_highest = locpot[-1].zheight - drop_angstroms_from_edges
  margin_to_get = within_angstroms + drop_angstroms_from_edges
  #Get all the locpots within five angstroms of either side
  pots_lower = [i.potential for i in locpot if i.zheight > z_lowest and i.zheight < margin_to_get]
  pots_upper = [i.potential for i in locpot if i.zheight < z_highest and i.zheight > z_highest - margin_to_get]
  if verbose == True and numpy.std(pots_lower) > 0.1:
    warnings.warn('The standard deviation {} of this vacuum level is high!'.format(numpy.std(pots_lower)))
    plot=True
  if plot:
    zheights, potential = zip(*locpot)
    plt.plot(zheights, potential)
    plt.show()
  return Vaclevel(numpy.mean(pots_lower), numpy.mean(pots_upper))

class CalcNotFinished(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

"""
Because you want the VBM of the bare slab to be aligned to zero...

These values are the vacLevelShifted(bare slab) = vaclevel(bare slab) - vbm(bare slab)

To align vacuum levels of adsorbed systems to these, do energies + shift

Where shift = vaclevel(bare slab) - vbm(bare slab) - vaclevel(adsys)

Where shift = vacLevelShifted(bare slab) - vaclevel(adsys)
"""
VAC_REF_ANATASE = 2.195 + 4.2749 #Three layer 2x5x3 from adchromes project
VAC_REF_RUTILE = (2.195 + 0.66) + 4.15 
VAC_REF_ANATASE_2L = 4.959 + 1.460 #Two layer anatase 2x5x2 from mchromes project
VAC_REF_TEST = 1.460 - - 4.959
ALL_VAC_REFS = {'VAC_REF_ANATASE': VAC_REF_ANATASE, 'VAC_REF_RUTILE': VAC_REF_RUTILE, 'VAC_REF_ANATASE_2L': VAC_REF_ANATASE_2L, 'VAC_REF_TEST': VAC_REF_TEST }


class LdosInfo(object):
  """
  This class holds all the information about an Ldos. 

  An ldos has multiple parts, such as the Anchor ldos, molecule ldos, Surface ldos. 
  These different parts are stored in self.pdos_collection, a dictionary where the keys are PdosParts and the values is a tuple of Pdos(dos_total, dos_s, dos_p, dos_d)
  """

  def __init__(self, calcpath=None, calcname=None, 
    vac_ref = None #Choose this from one of the above VAC_REF_* values
    ):
      self.adsys = None
      self.fermi_level_index = None 
      self.nelect = None
      self.band_gap = None
      self.calcname = calcname
      self.calcpath = None
      self.conduction_e = None
      self.dos_data = None
      self.dye_gap = None
      self.dye_homo = None
      self.dye_homo_index = None
      self.dye_homo_height = None
      self.dye_lumo = None
      self.dye_lumo_index = None
      self.dye_lumo_height = None
      self.anchor_lumo = None 
      self.anchor_lumo_index = None 
      self.anchor_lumo_height  = None 
      self.core_lumo = None #corresponds to photocore
      self.core_lumo_index = None #corresponds to photocore
      self.core_lumo_height  = None #corresponds to photocore
      self.core_homo = None #corresponds to photocore
      self.core_homo_index = None #corresponds to photocore
      self.core_homo_height  = None #corresponds to photocore
      self.subplot = None
      self.num_plot_labels = 0 #Keeps track of how many labels you've already added
      self.kp = None
      self.locpot = None
      self.num_charge_transfer_states_total = None
      self.num_charge_transfer_states_anchor = None
      self.num_dye_states_in_gap = None
      self.num_dye_virtual_peaks = None
      self.outcar = None
      #Note that when properly aligned, this value should equal the shifted vacuum level of bare slab 
      # = vaclevel(bare slab) - vbm(bare slab), and should be the same across all calcs.
      self.vac_level_lower_shifted = None
      self.vac_level_lower = None
      self.vac_level_lower_reference = None
      if vac_ref != None: self.set_reference_vac(vac_ref)
      self.vac_level_upper = None
      self.vac_level_source = None #Value from locpot or from outcar?
      self.valence_e = None
      #Stores the pdos of various parts of the system, eg Surface, Anchor, molecule
      self.pdos_collection = {}
      if calcpath:
        self.calcpath = calcpath
        if (calcname):
          self.calcname == calcname
        else:
          self.calcname = os.path.basename(calcpath)
        self.initialize_dos_info(calcpath)


  def print_info(self, with_calcpath=False):
    """
    If the flag with_calcpath is True, then it also prints the absolute calcpath as a column. For interactive usage, this should be false as it takes up a lot of room.

    >>> info = dte_dos.print_info(with_calcpath=False)
    >>> outputfile = join(TESTDIR, "print_info_dte.dat")
    >>> with open(outputfile, 'w') as datfile: datfile.writelines(info)

    >>> info = ana_dte_dos.print_info(with_calcpath=False)
    >>> outputfile = join(TESTDIR, "print_info_ana_dte.dat")
    >>> with open(outputfile, 'w') as datfile: datfile.writelines(info)
    """
    from vasptasks import get_info_from_calcname
    phase, mol, state = get_info_from_calcname(self.calcname)
    keys_of_interest = ['nelect', 'band_gap', 'conduction_e', 'dye_gap', 'dye_homo', 'dye_homo_height', 'dye_lumo', 'dye_lumo_height', 'core_lumo', 'core_lumo_height', 'core_homo', 'core_homo_height', 'kp', 'num_charge_transfer_states_total', 'num_charge_transfer_states_anchor', 'num_dye_states_in_gap', 'oxygen1s_energy_unshifted', 'valence_e', 'vac_level_lower', 'vac_level_upper', 'vac_level_lower_shifted']
    infolines = []
    for key in keys_of_interest:
      value = self.__dict__.get(key, None) 
      if value != None:
        infolines.append(INFO_FORMAT.format(self.calcname, phase, mol, state, key, value))
    return infolines


  def initialize_dos_info(self, calcpath):
    """ This method is called upon initializing LdosInfo with a calcpath """
    from vasptasks import get_number_of_electrons, is_finished
    self.outcar=join(calcpath, 'OUTCAR')
    if not is_finished(self.outcar):
      self = None
      raise CalcNotFinished("This calc isn't finished!")

    if 'LOCPOT_Z' in os.listdir(calcpath):
      self.locpot = join(calcpath, 'LOCPOT_Z')

    doscar=join(calcpath, 'DOSCAR')
    if not os.path.isfile(self.outcar) or not os.path.isfile(doscar):
      self = None
      raise IOError("Couldn't find DOSCAR. {} doesn't appear to be a valid VASP calcpath.".format(calcpath))
    # Check size of Doscar to see if it's bigger than one megabyte
    if os.stat(doscar).st_size / 1e6 < 1: 
      raise IOError("DOSCAR doesn't seem to be valid: {}".format(doscar))

    from adsorbed_system import AdsorbedSystem
    adsys = AdsorbedSystem(calcpath)
    self.adsys = adsys
    self.nelect = get_number_of_electrons(self.outcar)
    self.dos_data = vcalc.VaspDos(doscar,)
    self.energies = self.dos_data.energy # This is the x-axis in a typical DOS plt.
    #Align energies and then begin analysis of dos
    self.set_fermi_level()
    # print('Katiter: Not aligning anything')
    self.get_vacuum_level_from_file()

    
    if self.adsys.has_surface:
      #Get info about the surface. Also, get the info about the oxygen 1s of TiO2 in order to shift all the energies
      pdos_surface = get_dos_of_these_atoms(adsys.indices_surface, self.dos_data)
      self.pdos_collection[PdosParts.Surface] = pdos_surface
      self.set_surface_info(pdos_surface)

    if self.adsys.has_molecule:
      #Get info about the molecule
      pdos_molecule = get_dos_of_these_atoms(adsys.indices_molecule, self.dos_data)
      self.pdos_collection[PdosParts.molecule] = pdos_molecule
      pdos_anchor = get_dos_of_these_atoms(adsys.indices_anchor, self.dos_data)
      self.pdos_collection[PdosParts.Anchor] = pdos_anchor
      indices_of_photocore = [i for i in adsys.indices_molecule if i not in adsys.indices_anchor]
      pdos_photocore = get_dos_of_these_atoms(indices_of_photocore, self.dos_data)
      self.pdos_collection[PdosParts.Photocore] = pdos_photocore

      self.dye_homo, self.dye_homo_index, self.dye_homo_height = self.get_homo(pdos_molecule)
      self.dye_lumo, self.dye_lumo_index, self.dye_lumo_height = self.get_lumo(pdos_molecule)
      self.core_homo, self.core_homo_index, self.core_homo_height = self.get_homo(pdos_photocore, min_homo_height=2)
      self.core_lumo, self.core_lumo_index, self.core_lumo_height = self.get_lumo(pdos_photocore, min_lumo_height=2)
      self.anchor_lumo, self.anchor_lumo_index, self.anchor_lumo_height = self.get_lumo(pdos_anchor)
      self.dye_gap = self.dye_lumo - self.dye_homo

    # if self.adsys.is_adsorbed:
    #   self.set_charge_transfer()



  def get_vacuum_level_from_file(self):
    if self.locpot != None:
      self.vac_level_lower, self.vac_level_upper = get_vaclevel_from_locpot(self.locpot)
      self.vac_level_source = "LOCPOT_Z"
    elif self.outcar != None:
      self.vac_level_lower, self.vac_level_upper = get_vaclevel_from_outcar(self.outcar)
      self.vac_level_source = "OUTCAR"
    if self.vac_level_lower == None:
      warnings.warn('No vacuum level found for this calculation. Energies not aligned.')
      return
    else:
      self.shift_energies_to_reference_vac()
      


  def set_reference_vac(self, ref_vac):
    if ref_vac == None: return
    if isinstance(ref_vac, str):
      self.vac_level_lower_reference = ALL_VAC_REFS.get(ref_vac, None)
    elif isinstance(ref_vac, float):
      self.vac_level_lower_reference = ref_vac
    else: 
      raise Exception('This is not a valid vacuum level. Energies not aligned. Choose one of ' + str(ALL_VAC_REFS.keys()))
    if self.vac_level_lower != None: 
      self.shift_energies_to_reference_vac()


  def shift_energies_to_reference_vac(self):
    if self.vac_level_lower_reference == None:
      warnings.warn('No reference vacuum level defined. Energies not aligned. Choose one of ' + str(ALL_VAC_REFS.keys()))
      return
    shiftamount = self.vac_level_lower_reference - self.vac_level_lower
    self.vac_level_lower_shifted = self.vac_level_lower + shiftamount
    self.energies = map(lambda x: x + shiftamount, self.energies)
    print("Energies now aligned to vacuum level from {} and reference vacuum {:.2f}".format(self.vac_level_source, self.vac_level_lower_reference))

  def set_fermi_level(self):
    dos = self.dos_data
    if dos == None:
      raise Exception("The self.dos_data must be initialized before setting the fermi level!")
    self.fermi_level_index = next(i[0] for i in zip(range(0, len(dos.energy)), 
      dos.integrated_dos, dos.dos) if i[1] >= self.nelect)

  def set_surface_info(self, pdos_surface):  
    """ Get width of band gap. Valence/conduction band energies are determined as 5% of the tallest p/d peaks respectively.
    """
    valence_cutoff = 0.05 * max(pdos_surface.p)
    conduction_cutoff = 0.05 * max(pdos_surface.d)
    self.valence_e, _ = next(i for i in reversed(zip(self.energies, pdos_surface.tot)[0:self.fermi_level_index]) if i[1] > valence_cutoff)
    self.conduction_e, _ = next(i for i in zip(self.energies, pdos_surface.tot)[self.fermi_level_index + 10:] if i[1] > conduction_cutoff)
    self.band_gap = self.conduction_e - self.valence_e      

  def get_homo(self, pdos_molecule, min_homo_height=0.9):
    if self.fermi_level_index == None:
      raise Exception("The self.fermi_level_index must be initialized before setting the homo!")
    end_of_occupied_states = next(index for (index, dyedos) in zip(range(0, len(self.energies)), pdos_molecule.tot) 
      if index > self.fermi_level_index + 1 and dyedos < 0.1)
    #Reverse the list so that the peak_detect look-ahead won't miss the homo peak
    occupied_energies = list(reversed(self.energies[: end_of_occupied_states]))
    pdos_molecule = self.pdos_collection.get(PdosParts.molecule, None)
    occupied_dos = list(reversed(pdos_molecule.tot[: end_of_occupied_states]))
    peaks = peak_detect(occupied_energies, occupied_dos,
      lookahead=2, delta=0.1)
    try: 
      dye_homo, dye_homo_height = next((x, height) for (x, height) in peaks if height > min_homo_height)
      dye_homo_index = next(index for index, energy in enumerate(self.energies) if energy >= self.dye_homo)
      return OrbitalInfo(dye_homo, dye_homo_index, dye_homo_height)
    except StopIteration:
      print("Couldn't find dye HOMO :(")


  def get_lumo(self, pdos_molecule, min_lumo_height=0.5):
    if self.fermi_level_index == None: 
      raise Exception("set_lumo needs the fermi_level_index!")
    virtual_energies = self.energies[self.fermi_level_index+1 :]
    virtual_pdos = pdos_molecule.tot[self.fermi_level_index+1 :]
    peaks = peak_detect(virtual_energies, virtual_pdos,
      lookahead=2, delta=0.1)
    try: 
      dye_lumo, dye_lumo_height = next((x, height) for (x, height) in peaks if height > min_lumo_height)
      dye_lumo_index = next(index for index, energy in enumerate(self.energies) if energy >= self.dye_lumo) # np.where(self.energies==self.dye_lumo)
      return OrbitalInfo(dye_lumo, dye_lumo_index, dye_lumo_height)
    except StopIteration:
      print("Couldn't find dye LUMO :(")


  def set_charge_transfer(self):
    """set Charge Transfer by getting the number of electrons in the charge transfer states, ie the virtual orbitals between fermi level and LUMO
    NOTE: Currently retired """
    if self.dye_lumo_index != None and self.dye_homo_index != None:
      pdos_dye = self.pdos_collection.get(PdosParts.molecule, None)
      pdos_anchor = self.pdos_collection.get(PdosParts.Anchor, None)
      if (pdos_dye == None or pdos_anchor == None):
        print("You have to put Anchor and molecule into the pdos_collection first!")
      pdos_dye = pdos_dye.tot
      pdos_anchor = pdos_anchor.tot
      #Get a point after the homo such that the dos_molecule is zero
      start_index = next(index for index, dosheight in enumerate(pdos_dye) if dosheight < 0.1 and index > self.dye_homo_index)
      end_index = self.dye_lumo_index
      # Compute the area using Simpson's rule
      from scipy.integrate import simps
      self.num_charge_transfer_states_total  = simps(pdos_dye[start_index:end_index], x=self.energies[start_index:end_index])
      self.num_charge_transfer_states_anchor  = simps(pdos_anchor[start_index:end_index], x=self.energies[start_index:end_index])

  def plot_this_dos():
    return

  def plot(self, plot_only_these_items=[0,4,5]):
    """
    By default, plots everything in .
    However, you can give a list integers corresponding to things to plot .
    These numbers are the zOrder un the enum list of PdosParts.

    >>> dte_dos.zoom_to_molecule([0, 3]) #Plots only the surface and photocore
    """
    self.fig = plt.figure()
    self.subplot = self.fig.add_subplot(111)
    # Get PdosParts in the order that they should be plotted
    sorted_elements_to_plot = self.pdos_collection.items();
    # sorted(self.pdos_collection.items(), 
    # key = lambda (pdos_part, pdos): pdos_part.value.zOrder)

    for pdos_part, pdos_spd in sorted_elements_to_plot:
      if plot_only_these_items == None or pdos_part.value.zOrder in plot_only_these_items:
        hex_color = '#' + pdos_part.value.color[2:]
        # if pdos_part == PdosParts.Surface_Oxygens:
        #   #Draw the oxygen 1s peak
        #   self.subplot.fill_between(self.energies, 0, pdos_spd.s, color=hex_color)
        # if pdos_spd.tot != None
        self.subplot.fill_between(self.energies, 0, pdos_spd.tot, color=hex_color)

  def hide_Y(self):
    #Hide y axis to look more professional
    plt.gca().get_yaxis().set_ticks([])

  def label_axes(self, xlabel='Energy (eV)', ylabel='PDOS'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

  def show(self, interactive = True):
    if self.subplot == None: self.plot()
    if interactive:
        plt.ion() #Interactive mode
    plt.show()

  def draw(self):
    """ Updates the plot after using, for instance, show_fermi"""
    if self.subplot == None: self.plot()
    plt.draw()

  def close(self):
    if self.subplot != None:
      plt.close()

  def save(self, outputfile="plot_DOS.pdf", size=(7,3)):
    if self.subplot == None: self.plot()
    plt.gcf().set_size_inches(size)
    plt.savefig(outputfile, bbox_inches='tight', dpi=100)
    #print "Plot drawn to {}".format(outputfile)


  def zoom(self, **kwargs):
    #Set plot bounds
    xmin = kwargs.get('xmin', self.energies[5])
    xmax = kwargs.get('xmax', self.energies[-5])
    ymax = kwargs.get('ymax', None)
    self.subplot.set_xlim([xmin, xmax])
    if (ymax != None):
      self.subplot.set_ylim([0, ymax])

  def zoom_to_molecule(self):
    """
    >>> dte_dos.zoom_to_molecule()
    >>> dte_dos.save(join(TESTDIR, "zoom_to_molecule_dte.pdf"))
    >>> ana_dte_dos.zoom_to_molecule()
    >>> ana_dte_dos.save(join(TESTDIR, "zoom_to_molecule_ana_dte.pdf"))
    """
    pdos_dye = self.pdos_collection.get(PdosParts.molecule, None)
    if pdos_dye == None: return
    ymax = max(pdos_dye.tot[5:-5]) * 1.0 #* 1.2
    self.subplot.set_ylim([0, ymax])
    plt.draw()



  def draw_line_at_energy(self, energy, name=None):
    """ Puts a line at the specified value. 
    Writes the values in the upper right corner if a name is supplied.
    """
    if self.subplot == None: self.plot()
    self.subplot.axvline(x=energy, color='DarkCyan', linestyle='dashed')
    if name != None:
      self.num_plot_labels = self.num_plot_labels + 1
      y_displace = 0.95 - self.num_plot_labels * 0.1
      self.subplot.annotate('{}={:0.2f}'.format(name, energy), 
              xy=(0.95, y_displace),  xycoords='axes fraction', size=12,
              horizontalalignment='right', verticalalignment='top')
    plt.draw()
    

  def show_fermi(self):
    fermi_level_energy = self.energies[self.fermi_level_index]
    self.draw_line_at_energy(fermi_level_energy)
    

  def show_homolumo(self):
    """ 
    >>> dte_dos.show_homolumo()
    >>> ana_dte_dos.show_homolumo()
    """
    if self.dye_homo != None:
      self.draw_line_at_energy(self.dye_homo, 'HOMO')
    if self.dye_lumo != None:
      self.draw_line_at_energy(self.dye_lumo, 'LUMO')
    if self.core_lumo != None:
      self.draw_line_at_energy(self.core_lumo, 'coreLUMO')

  def annotate_lumos(self, lookahead=2, delta=0.1):
    """
    >>> dte_dos.annotate_lumos()
    >>> ana_dte_dos.annotate_lumos()
    """
    if self.subplot == None: self.plot()
    pdos_molecule = self.pdos_collection.get(PdosParts.molecule, None)
    if pdos_molecule == None: return #Probably just a lone surface
    virtual_energies = self.energies[self.fermi_level_index+1 :]
    virtual_pdos = pdos_molecule.tot[self.fermi_level_index+1 :]
    peaks = peak_detect(virtual_energies, virtual_pdos, lookahead=lookahead, delta=delta)
    maxPeaks = 5
    currentPeaks = 0
    for x, height in peaks:
      self.subplot.plot(x, height, 'bo')

      currentPeaks = currentPeaks + 1
      if currentPeaks > maxPeaks: break
    plt.draw()


  def write_name(self):
    if self.subplot == None: self.plot()
    self.subplot.annotate(self.calcname,  
              xy=(0.05, 0.95),  xycoords='axes fraction', size=12,
              horizontalalignment='left', verticalalignment='top')
    plt.draw()
    

def _test(verbose=None):
  import doctest
  if not os.path.exists(TESTDIR):
    os.makedirs(TESTDIR)
  # Define LdosInfo objects to use throughout doctesting
  dte_dos = LdosInfo(asethings.DTE_CLOSED_PATH)
  ana_dte_dos = LdosInfo(asethings.ANA_AZB_TRANS_PATH)
  extraglobs = {"TESTDIR": TESTDIR, 'dte_dos': dte_dos, 'ana_dte_dos': ana_dte_dos}
  doctest.testmod(verbose=verbose, extraglobs=extraglobs)


def _main(argv=None):
  return 0

if __name__ == "__main__":
    progname = sys.argv[0]
    usage = """usage: %prog [options] incar param new_value 

    This script changes a value of an INCAR tag to a certain value. 
    Eg. %prog ./INCAR ALGO Normal""".replace("%prog", progname)    
    parser = OptionParser(usage=usage)
    parser.add_option('--profile', '-P',
                       help    = "Print out profiling stats",
                       action  = 'store_true')
    parser.add_option('--test', '-t',
                       help   ='Run doctests',
                       action = 'store_true')
    parser.add_option('--verbose', '-v',
                       help   ='print debugging output',
                       action = 'store_true')
    global options
    (options, args) = parser.parse_args()

    if options.test:
        _test(verbose=options.verbose)
        exit()

    sys.exit(_main(args))


