#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2014-12-05 by Kathy Chen

Gathers all the useful constants you have used so far.
"""
from ase.units import mol, kJ, kcal, Hartree, Rydberg
import os, sys
import ase.io
import re

CHEMICAL_SYMBOLS = ['X',  'H',  'He', 'Li', 'Be',
                    'B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si',
                    'P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se',
                    'Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                    'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe',
                    'Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                    'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W',
                    'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                    'At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu',
                    'Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr']

# chiral_system.py:
# CUTOFF_DICT = {'S': 0.8, 'Se': 1.0, 'Cu': 1.1, 'Ti': 1.1, 'Zn': 1.2, 'Ca': 1.7, 'C': 0.6, 'O': 0.5, 'Li': 1.3}

CUTOFF_DICT = {'S': 0.7, 'C': 0.6, 'Ti': 1.1, 'O': 0.5}


DIST_RUT_2PLANES = 2.58
DIST_RUT_4PLANES = 2.9
DIST_RUT_5PLANES = 4.6
DIST_ANA_3PLANES = 2.40 #1 layers fixed
DIST_ANA_4PLANES = 3.20 #1 layers fixed
DIST_CUSCN_3PLANES = 3.0 #1 layer but not the sulfur fixed
DIST_CUSCN_4PLANES =  4.0 #1 layer fixed

DIST_CUSCN_110_2PLANES = 2.5
DIST_CUSCN_100_2PLANES = 2.5
DIST_CUSCN_101_1LAYER = 2.7
DIST_CUSCN_001_2PLANES = 2.5
DIST_CUSCN_001_4PLANES = 4.8
#Load some sample geoms
from os.path import join

TEST_DIR = join(os.path.dirname(os.path.realpath(__file__)), "test")

## Handy for testing, this one loads faster because it's gamma grid with 300 dos points
ANA_AZB_TRANS_SMALL_PATH = join(TEST_DIR, "ana_azb_trans_small")
ANA_AZB_TRANS_SMALL = ase.io.read(join(ANA_AZB_TRANS_SMALL_PATH, "CONTCAR"))

## Some lone photochromes
AZB_TRANS_PATH = join(TEST_DIR, "azb_trans")
DTE_CLOSED_PATH = join(TEST_DIR, "dte_closed")
AZB_TRANS = ase.io.read(join(AZB_TRANS_PATH, "CONTCAR"))
DTE_CLOSED = ase.io.read(join(DTE_CLOSED_PATH, "CONTCAR"))

## Some photochromes + surfaces
ANA_AZB_TRANS_PATH = join(TEST_DIR, "ana_azb_trans")
ANA_DTE_CLOSED_PATH = join(TEST_DIR, "ana_dte_closed")
ANA_AZB_TRANS = ase.io.read(join(ANA_AZB_TRANS_PATH, "CONTCAR"))
ANA_DTE_CLOSED = ase.io.read(join(ANA_DTE_CLOSED_PATH, "CONTCAR"))


def convert_number_into_symbol(atomic_number):
  return CHEMICAL_SYMBOLS[int(atomic_number)]


def get_calc_from_com(comfile, verbose=False):
  """ Note: You made this work for espresso .inp files too. """
  geomlines = []
  tempfile = 'temp.xyz'
  patternGeom = re.compile("\s*(\S{1,2})\s+(-?[0-9]+.[0-9]{3,})\s+(-?[0-9]+.[0-9]{3,})\s+(-?[0-9]+.[0-9]{3,}).*")
  with open(comfile, 'r') as file:
    for line in file:
      foundGeomGroup = re.match(patternGeom, line)
      if foundGeomGroup:
        groups = list(foundGeomGroup.groups())
        if verbose: print(groups)
        if not groups[0].isalpha():
          groups[0] = asethings.convert_number_into_symbol(groups[0])
        geomlines.append(' \t '.join(groups) + '\n')
  with open(tempfile, "w") as file:
    file.write('{}\n\n'.format(len(geomlines)))
    file.writelines(geomlines)
  calc = ase.io.read(tempfile)
  os.remove(tempfile)
  return calc


def convert_kjmol_ev(value):
  return value * (kJ/mol)

def convert_ev_kjmol(value):
  return value / (kJ/mol)

def convert_kcal_ev(value):
  return value * (kcal/mol)

def convert_ev_kcal(value):
  return value / (kcal/mol)

def convert_au_ev(value):
  return value * Hartree

def convert_ev_au(value):
  return value / Hartree

def convert_nm_ev(value):
  return 1239.84193 / value #hc/nm

def convert_ev_nm(value):
  return 1239.84193 / value

def convert_ryd_ev(value):
  return value * Rydberg

def convert_ev_ryd(value):
  return value / Rydberg

#For surface energy
def convert_evA_Jm2(value):
  return value / Rydberg