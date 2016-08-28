#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 2015 by Kathy Chen

Uses pandas and pymatgen.io.vasp to visualize the electronic structure of solid state systems. Using the result of a bandstructure calculation (VASP and Quantum Espresso) and visualize the 4-D result in 2-D.
"""

import sys
import matplotlib as mpl
import pymatgen.io.vasp as pyvasp
from optparse import OptionParser
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.electronic_structure.core import Spin, Orbital
import ase.io
from os.path import join
Interest = namedtuple('Interest', 'label color pdos')


class BandStructure(object):

    def __init__(self, calcdir, verbose=True, read_bands=True, save_pcalc=False):
        """ Note that for memory reasons we do not store pcalc or procar as instance variables.
        """
        self.calcdir = calcdir
        self.calc = ase.io.read(join(calcdir, 'CONTCAR'))
        if verbose: sys.stdout.write('Reading vasprun.xml and bands... ')
        pcalc = pyvasp.Vasprun(join(calcdir, 'vasprun.xml'))
        if save_pcalc: self.pcalc = pcalc
        self.band_gap, self.vbm, self.cbm, self.is_direct = pcalc.eigenvalue_band_properties
        if read_bands:
            self.bands = pcalc.get_band_structure(join(calcdir, 'KPOINTS'), line_mode=True)
            self.nkpts = len(self.bands.kpoints)
            self.nbands = self.bands.nb_bands

            if verbose: sys.stdout.write('Reading procar... ')
            procar = pyvasp.Procar(join(calcdir, 'PROCAR'))  # projected bands
            self.bigprocar = self._get_pandas_from_procar(procar)

        if verbose: sys.stdout.write('Converting to Pandas... ')
        self.bigdos = self._get_bigdos_from_pcalc(pcalc)
        self.totdos = self.get_dos_contrib()  # Get the total dos

        #Other useful information
        self.natoms = self.calc.get_number_of_atoms()
        self.unscaled_energies = pcalc.tdos.energies
        self.scaled_energies = pcalc.tdos.energies - pcalc.efermi
        self.emin = self.scaled_energies[0]
        self.emax = self.scaled_energies[-1]
        self.efermi = pcalc.complete_dos.efermi
        self.energy = pcalc.final_energy
        if verbose: sys.stdout.write('DONE! :)')

    def _get_df_of_one_atom(self, procar, atomno):
        """Convert procar.data to pandas table!"""
        sdata = procar.data[atomno]
        nkpts = len(sdata)
        all_dfs = map(lambda kpt: pd.DataFrame(sdata[kpt].get('bands')), range(1, nkpts+1))
        return pd.concat(all_dfs, keys=range(1, nkpts+1), names=['kpt', 'orbital'])

    def _get_pandas_from_procar(self, procar):
        atom_keys = range(0, len(procar.data))
        all_atoms = map(lambda a: self._get_df_of_one_atom(procar, a), atom_keys)
        bigprocar = pd.concat(all_atoms, keys=atom_keys, names=['atomno'])
        return bigprocar

    def _get_df_of_one_atomdos(self, pcalc, atomno, orbs=Orbital.all_orbitals[0:9]):
        """
        :Example:

        get_df_of_one_atomdos(pcalc, 1)
        """
        orbitals = map(lambda orb: pcalc.pdos[atomno].get(orb)[Spin.up], orbs)
        return pd.DataFrame(np.asarray(orbitals).transpose(), columns=map(str, orbs), index=pcalc.tdos.energies)

    def _get_bigdos_from_pcalc(self, pcalc):
        atom_keys = range(0, len(pcalc.pdos))
        bigdos = pd.concat(map(lambda a: self._get_df_of_one_atomdos(pcalc, a), atom_keys), keys=atom_keys, names=['atomno', 'energy'])
        return bigdos

    def _readable_orbitals(self, orbitals):
        """ This convenience method is for your `get_dos_contrib` and `get_procar_contrib` methods.
        """
        all_orbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dx2', 'dz2', 'dxz']
        if isinstance(orbitals, (list, tuple)):
            assert all([i in all_orbitals for i in orbitals])
            return orbitals
        if orbitals is None or orbitals == slice(None):
            return slice(None)
        if orbitals == 's':
            return ['s']
        elif orbitals == 'p':
            return ['px', 'py', 'pz']
        elif orbitals == 'd':
            return ['dxy', 'dyz', 'dx2', 'dz2', 'dxz']
        elif orbitals == 'eg':
            return ['dz2', 'dx2']
        elif orbitals == 't2g':
            return ['dxy', 'dyz', 'dxz']
        else:
            raise Exception('What kind of orbital is this?' + str(orbitals))

    def _get_selected_atoms(self, selected_atoms):
        """
        What atoms do you want to select from your system? You can give it ['Cu'], ['Cu', 28, 4], [3, 2, 'Ti', 'O'] and this method selects out the right atomic indices for you.
        Note, the argument always has to be a list!
        """
        if selected_atoms is None or selected_atoms == slice(None) or len(selected_atoms) == 0:
            return slice(None)
        else:
            atomnos = [i for i in selected_atoms if isinstance(i, int)]
            atomnames = [i for i in selected_atoms if isinstance(i, str)]
            new_selected_atoms = atomnos + ([a.index for a in self.calc if a.symbol in atomnames])
            return new_selected_atoms

    def get_procar_contrib(self, selected_atoms=slice(None), selected_orbs=slice(None), by_proportion=True):
        """ For each band and each kpoint, get the percentage of contribution.
        :param selected_orbitals: subset of ['s', 'px','py','pz', 'dxy', 'dyz', 'dx2', 'dz2', 'dxz']

        :Example:

        contrib_pxpy = get_procar_contrib(bigprocar, selected_orbs = ['px','py'], by_proportion=False)
        contrib_d = get_procar_contrib(bigprocar, selected_orbs = ['dxy', 'dyz', 'dx2', 'dz2', 'dxz'])
        """
        selected_atoms = self._get_selected_atoms(selected_atoms)
        selected_orbs = self._readable_orbitals(selected_orbs)
        df_contrib = self.bigprocar.loc[(selected_atoms, slice(None), selected_orbs), :].groupby(level=['kpt']).sum()
        if by_proportion:
            df_tot = self.bigprocar.loc[(slice(None), slice(None), slice(None)), :].groupby(level=['kpt']).sum()
            return (df_contrib/df_tot).fillna(0.0)
        else:
            df_contrib = df_contrib.fillna(0.0)
            df_contrib = df_contrib / np.max(df_contrib.values)
            return df_contrib

    def get_dos_contrib(self, selected_atoms=slice(None), selected_orbs=slice(None), clean=True):
        """Get the PDOS of only selected atoms and selected orbitals. Or none, if you want the total dos.
        :param selected_orbitals: subset of ['s', 'px','py','pz', 'dxy', 'dyz', 'dx2', 'dz2', 'dxz']
        """
        selected_atoms = self._get_selected_atoms(selected_atoms)
        selected_orbs = self._readable_orbitals(selected_orbs)
        dos_contrib = self.bigdos.loc[(selected_atoms, selected_orbs)].groupby(level=['energy']).sum().sum(axis=1)
        if clean:  # Clean the weird huge numbers at the beginning and end of the DOS.
            dos_contrib[0] = 0
            dos_contrib[-1] = 0
        return dos_contrib

    def _rgline(self, ax, k, e, color=None, weight=None, colormap=mpl.cm.summer, 
                bs_linewidth=1.0, bs_linewidth_fattener=0.5):
        pts = np.array([k, e]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        nseg = len(k)-1
        if color is None:
            c = ['#000000'] * len(k)
        else:
            c = colormap([0.5*(color[i]+color[i+1]) for i in range(nseg)])
        if weight is None:
            w = [bs_linewidth] * len(k)
        else:
            w = [bs_linewidth * (weight[i] + weight[i + 1]) + bs_linewidth_fattener for i in range(nseg)]
        lc = mpl.collections.LineCollection(seg, colors=c, linewidth=w)
        ax.add_collection(lc)

    def plot_bandstructure_dos(self, dos_contrib_tot=None,
                               title=None, verbose=True,
                               outfile=None, show=True,
                               xlabel=r"k-points", ylabel=r"$E-E_f$ (eV)",
                               special_pts=None,
                               color_tot_dos="#D0D0D0",
                               figsize=(6, 7), width_ratios=[1.8, 1],
                               emin=None, emax=None, pdos_max=None,
                               legendsize=10, fontsize=14, legend=True,
                               fill_pdos_lines=False, show_pdos=True,
                               color=None, weight=None, interests=[], **kwargs
                               ):
        if dos_contrib_tot is None:
            dos_contrib_tot = self.totdos
        font = {'size': fontsize, 'serif': ['computer modern roman']}  # This is to prevent bolding of the axes
        plt.rc('font', **font)
        plt.figure(figsize=figsize)
        if show_pdos:
            gs = mpl.gridspec.GridSpec(1, 2, width_ratios=width_ratios)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])  # sharey=ax1)
        else:
            ax1 = ax1 = plt.gca()

        if emin is None: emin = self.emin - 0.5
        if emax is None: emax = self.emax + 0.5
        if pdos_max is None:
            pdos_max = max(dos_contrib_tot)
        if verbose: print("Setting PDOS max to {:.1f} (change pdos_max if you want a different value)".format(pdos_max))
        for b in range(self.nbands):
            e_along_kpts = [e - self.bands.efermi for e in self.bands.bands[Spin.up][b]]
            c = None if color is None else color[b+1].values
            w = None if weight is None else weight[b+1].values
            self._rgline(ax1, range(self.nkpts), e_along_kpts, color=c, weight=w, **kwargs)

        #Plot the total DOS again as a line so it shows up in the legend
        # ax2.plot(dos_contrib_tot, self.scaled_energies, color=color_tot_dos, label="Total")
        if verbose: print("You have {} kpoints in your path and {} specified special_pts, with increments of {} kpoints between each special point".format(self.nkpts, len(special_pts), self.nkpts/(len(special_pts)-1)))
        if special_pts:
            tick_locations = range(0, self.nkpts+1, self.nkpts/(len(special_pts)-1))
            tick_locations[-1] = tick_locations[-1]-1  # To handle the kpoint fencepost
            ax1.set_xticks(tick_locations)
            ax1.set_xticklabels(special_pts)

        # Set up the style
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid()
        ax1.set_xlim(0, self.nkpts - 1)  # To handle the kpoint fencepost
        ax1.set_ylim(emin, emax)
        [i.set_linewidth(0.5) for i in ax1.spines.itervalues()]
        [i.set_color((0.3, 0.3, 0.3)) for i in ax1.spines.itervalues()]
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.grid(linestyle='solid', alpha=0.1, linewidth=0.5)

        if show_pdos:
            # Plot the total dos. Note that emax here is how high the filling goes on the Y axis
            ax2.fill_between(dos_contrib_tot, self.emax, self.scaled_energies, color=color_tot_dos, facecolor=color_tot_dos, linewidth=0.5)

            # Plot the PDOS contributions
            for interest in interests:
                ax2.plot(interest.pdos.values, self.scaled_energies, color=interest.color, label=interest.label, linewidth=0.7)
                if fill_pdos_lines:
                    ax2.fill_between(interest.pdos.values, self.emax, self.scaled_energies, color=interest.color, facecolor=interest.color, linewidth=0.5)

            ax2.set_ylim(emin, emax)
            ax2.grid()
            ax2.set_xlim(1e-6, pdos_max)
            ax2.set_xlabel("PDOS")
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            [i.set_linewidth(0.5) for i in ax2.spines.itervalues()]
            [i.set_color((0.3, 0.3, 0.3)) for i in ax2.spines.itervalues()]
            ax2.grid(linestyle='solid', alpha=0.1, linewidth=0.5)
            ax2.yaxis.set_tick_params(size=0)
            if legend:
                ax2.legend(fancybox=True, shadow=False, prop={'size': legendsize})

        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        if outfile is not None:
            plt.savefig(outfile)
        if show:
            plt.show()


    def plot_pdos_lines(self, dos_contrib_tot=None, outfile=None, figsize=(6, 3.5),
                        xlabel=r"$E-E_f$ (eV)", ylabel=r"PDOS",
                        color_tot_dos="#D0D0D0",
                        xmin=None, xmax=None,
                        ymin=0, ymax=None,
                        stagger_height=2, initial_stagger=0,
                        fontsize=14, interests=[], show=True,
                        adjust_right_margin_by=0.93,
                        fill_pdos_lines=False,
                        annotate_names=True):

        if dos_contrib_tot is None:
            dos_contrib_tot = self.totdos
        if xmin is None:
            xmin = self.emin
        if xmax is None:
            xmax = self.emax
        if ymax is None:
            ymax = max(dos_contrib_tot) + initial_stagger + len(interests) * stagger_height
        plt.figure(figsize=figsize)
        font = {'size': fontsize, 'serif': ['computer modern roman']}  # This is to prevent bolding of the axes
        plt.rc('font', **font)
        ax1 = plt.gca()
        stagger = initial_stagger + len(interests) * stagger_height
        ax1.fill_between(self.scaled_energies, stagger, dos_contrib_tot + stagger, color=color_tot_dos, label="Total", linewidth=0.1)
        if annotate_names: plt.annotate('Total', xy=(xmax, stagger + 0.02*ymax), color=color_tot_dos)

        for interest in reversed(interests):
            stagger = stagger - stagger_height
            ax1.plot(self.scaled_energies, interest.pdos.values + stagger, color=interest.color, label=interest.label, linewidth=0.7)
            if fill_pdos_lines: ax1.fill_between(self.scaled_energies, stagger, interest.pdos.values + stagger, color=interest.color, label=interest.label, linewidth=0.1)
            if annotate_names: plt.annotate(interest.label, xy=(xmax, stagger + 0.02*ymax), color=interest.color)
            # stack = stack + stagger_height
        # Adjust plot style
        ax1.axes.get_yaxis().set_ticks([])
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        #Make it look more professional
        [i.set_linewidth(0.5) for i in ax1.spines.itervalues()]
        [i.set_color((0.5, 0.5, 0.5)) for i in ax1.spines.itervalues()]
        ax1.yaxis.set_tick_params(size=0)
        ax1.set_yticklabels([])
        ax1.xaxis.set_ticks_position('bottom')
        #ax1.grid(linestyle='solid', alpha=0.1, linewidth=0.5)
        plt.tight_layout()
        plt.subplots_adjust(right=adjust_right_margin_by)
        if outfile:
            plt.savefig(outfile)
        if show:
            plt.show()
        else:
            plt.close();

    def view(self):
        ase.visualize.view(self.calc)

    def pickle_me(self, filename, compact=True):
        import jsonpickle
        # Note that this outputs much bigger file!
        # jsonpickle.set_encoder_options('simplejson', sort_keys=False, indent=4) 
        frozen = jsonpickle.encode(self)
        with open(filename, 'w') as f:
            f.write(frozen)
        print('Written to ' + filename)


def __test_pickle():
    #bandstructure = reload(bandstructure);
    from bandstructure import BandStructure
    bs = BandStructure('~/Projects/copper_serious/metallicize_001/copper_001_vac_0.00')
    bs.pickle_me('~/Projects/lork6.pickle')

def _test(verbose=None):
    """
    To interact with locals at any point:
    import interlude
    interlude.interact(locals())
    """
    import doctest
    doctest.testmod(verbose=verbose, extraglobs=None, optionflags=doctest.ELLIPSIS)
    print('Testing! :)')
    mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')
    mpl.rcParams['xtick.direction'] = 'out'
    #__test_plot_bandstructure('~/Projects/copper_001_vac_0.00')
    __test_plot_bandstructure(None)
    #__test_plot_pdos_lines_of_bulk(None)


def __test_plot_bandstructure(calcdir):
    truncated_hot = truncate_colormap(mpl.cm.afmhot, 0.0, 0.7)
    for vacuum in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        copper_metal = BandStructure('thesising/copper_001_metallization/copper_001_vac_{:.1f}'.format(vacuum))
        copper_contrib_color = copper_metal.get_procar_contrib(selected_atoms=['C', 'N', 'S'])
        copper_contrib_weight = copper_metal.get_procar_contrib(selected_atoms=['Cu'], selected_orbs='s')
        outfile = 'thesising/_001_vac_{:.1f}.png'.format(vacuum)
        copper_metal.plot_bandstructure_dos(color=copper_contrib_color, weight=copper_contrib_weight, colormap=truncated_hot, pdos_max=5, emin=-10, emax=10,
                                           special_pts=[r'$\Gamma$', r'$X$', r'$R$', r'$Z$', r'$\Gamma$', r'$M$', r'$A$', r'$Z$'],
                                           legendsize=12, fontsize=16, figsize=(3, 5), show_pdos=False,
                                           outfile = outfile, xlabel='', ylabel='', show=False)
        print("Made " + outfile)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    cmap = plt.get_cmap('jet')
    new_cmap = truncate_colormap(cmap, 0.2, 0.8)

    # To preview the old and new colormaps:
    arr = np.linspace(0, 50, 100).reshape((10, 10))
    cmap = plt.get_cmap('jet')
    new_cmap = truncate_colormap(cmap, 0.2, 0.8)
    ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
    ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
    plt.show()
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def _main(args):
    print("hi world")
    return 0


if __name__ == "__main__":
    progname = sys.argv[0]
    usage = """usage: %prog [options] file

    This script has two functions

    Eg. %prog file
    """.replace("%prog", progname)

    parser = OptionParser(usage=usage)
    parser.add_option(
        '--test', '-t',
        help='Run doctests',
        action='store_true')
    parser.add_option(
        '--verbose', '-v',
        help='print debugging output',
        action='store_true')
    parser.add_option(
        '--source', '-s',
        help="Include")
    (options, args) = parser.parse_args()

    if options.test:
            _test(verbose=options.verbose)
            exit()

    sys.exit(_main(args))
