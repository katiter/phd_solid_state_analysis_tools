#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 2015 by Kathy Chen
"""

import sys, os
from optparse import OptionParser
import ldos

def _test(verbose=None):
    import doctest
    doctest.testmod(verbose=verbose)

def _main(args):
    dos = ldos.LdosInfo(os.getcwd(), vac_ref=options.align)
    print "".join(dos.print_info(with_calcpath=False))
    dos.show_homolumo()
    dos.annotate_lumos()
    dos.write_name()
    dos.show(interactive=False)
    return 0

if __name__ == "__main__":
    progname = sys.argv[0]
    usage = """usage: %prog [options]

    When run in a VASP calcfolder, this script outputs a plot of the partial density of states.

    Eg. %prog """.replace("%prog", progname)    
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
    parser.add_option('--align', '-a',
                       help   ='Which vacuum level to align to. Choose one of ' + str(ldos.ALL_VAC_REFS.keys()),
                       default = None)
    (options, args) = parser.parse_args()

    if options.test:
        _test(verbose=options.verbose)
        exit()

    sys.exit(_main(args))

