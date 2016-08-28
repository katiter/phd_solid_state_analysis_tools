#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 2015 by Kathy Chen
"""

import sys
from adsorbed_system import AdsorbedSystem
from optparse import OptionParser
import os

TEST_DIR = "./test/ana_dte_closed"

def _test(verbose=None):
  import doctest
  doctest.testmod(verbose=verbose, extraglobs=None, optionflags=doctest.ELLIPSIS)

def read_avf(filename):
  """
  Reads in the ACF.dat file and returns the contents

  >>> data = read_avf(os.path.join(TEST_DIR, "ACF.dat"))
  >>> data[-1].charge
  1.3983
  """
  with open(filename, 'r') as f:
    data = f.readlines() 
    data = map(lambda line: line.split(), data[2:len(data)-4]) #exclude header, footer
    types = [int, float, float, float, float, float, float] #Convert str into numbers  
    data = map(lambda line: [type(x) for type, x in zip(types, line)], data)
    from collections import namedtuple
    AcfEntry = namedtuple('AcfEntry', 'atomno x y z charge min_dist volume')
    data = map(lambda line: AcfEntry(* line), data)
    return data

def get_charge_transfer(adsys, baderdata):
  charge_photochrome = sum([line.charge for index, line in enumerate(baderdata) if index in adsys.indices_photochrome])
  charge_surface = sum([line.charge for index, line in enumerate(baderdata) if index in adsys.indices_surface])
  # print "Total charge:\t{}".format(all_charges)
  # print "Surface charge:\t{}".format(charge_surface)
  # print "Photochrome charge:\t{}".format(charge_photochrome)
  # Calculate the canonical valence of the surface
  num_Ti = adsys.calc.get_chemical_symbols().count('Ti')
  surface_valence = num_Ti * 4 + num_Ti * 2 * 6
  return charge_surface - surface_valence

def _main(args):
  """
  >>> contcar = os.path.join(TEST_DIR, "CONTCAR")
  >>> acf = os.path.join(TEST_DIR, "ACF.dat")
  >>> adsys = AdsorbedSystem(contcar)
  >>> baderdata = read_avf(acf)
  """
  if not os.path.exists(args[0]):
    print "This CONTCAR file doesn't exist"
    return
  if not os.path.exists(args[1]):
    print "This bader ACF.dat file doesn't exist."
    return
  adsys = AdsorbedSystem(args[0])
  baderdata = read_avf(args[1])
  from vasptasks import get_info_from_calcname
  phase, mol, state = get_info_from_calcname(adsys.calcpath)
  value = get_charge_transfer(adsys, baderdata)
  print "{:<10}{:<15}{:<10}{:>15}".format(phase, mol, state, value)
  return 0

if __name__ == "__main__":
  progname = sys.argv[0]
  usage = """usage: %prog [options] CONTCAR ACF.dat

  This script takes in a vasp geometry file (CONTCAR) and the result of the bader tool (ACF.dat containing the atomic charges), separates the charges, and prints out three numbers: sum of the photochrome charge, sum of the surface charge, and the charge difference.

  Or it just prints the charge transferred to the surface.
  """.replace("%prog", progname)    
  parser = OptionParser(usage=usage)
  parser.add_option('--test', '-t',
                     help   ='Run doctests',
                     action = 'store_true')
  parser.add_option('--verbose', '-v',
                     help   ='print debugging output',
                     action = 'store_true')
  (options, args) = parser.parse_args()
  if len(args) == 0:
    args = [os.getcwd(), 'ACF.dat']

  if options.test:
      _test(verbose=options.verbose)
      exit()

  sys.exit(_main(args))

