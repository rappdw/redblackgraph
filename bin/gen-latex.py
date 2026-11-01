#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to convert")
    parser.add_argument("-p", "--preserve", help="Preserve intermediate files", action="store_true")
    args = parser.parse_args()

    # first go through the python notebook and switch out html "<img src"
    # with markdown "![..."
    dir, out_file_name = os.path.split(args.file)
    out_file_name = os.path.join(dir, "tmp." + out_file_name)
    with open(args.file, 'r', encoding='utf8') as file:
        with open(out_file_name, 'w', encoding='utf8') as out_file:
            for line in file:
                if line.strip().startswith('"<img src'):
                    line = line.replace("<img", "<!-- <img").replace('"/>\\n', '"/> -->\\n')
                elif line.strip().startswith('"<!-- !['):
                    line = line.replace('<!-- ', '').replace(' -->', '')
                out_file.write(line)

    # generate the latex file
    cmd = ["jupyter",
           "nbconvert",
           out_file_name,
           "--to",
           "latex"
           ]
    rc = subprocess.check_call(cmd, cwd=os.getcwd())
    # rc = 0
    # out_file_name = "/Users/drapp/sandbox/redBlackGraph/notebooks/tmp.Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.tex"
    if rc == 0:
        # Modify the tex file to fix up a few things that nbconvert doesn't quite handle
        # Fixup title:          \title{tmp.Red Black Graph - A DAG of Multiple, Interleaved Binary Trees}
        # Insert author:        \author{Daniel W Rapp}
        # Remove munged up \includegraphics command:
        #                       \let\Oldincludegraphics\includegraphics
        #                       % Set max figure width to be 80% of text width, for now hardcoded.
        #                       \renewcommand{\includegraphics}[1]{\Oldincludegraphics[width=.8\maxwidth]{#1}}
        # Add float package:    \userpackage{float}
        # Scale images:         \includegraphics[scale=0.3]
        # Figure placement:     \begin{figure}[H]
        tex_input = os.path.splitext(out_file_name)[0] + '.tex'
        dir, tex_output = os.path.split(tex_input)
        tex_output = os.path.join(dir, tex_output.replace("tmp.", ""))
        print(f"Editing latex in {tex_input}. Output: {tex_output}")
        # now go through the tex file and fix up a number of things
        skip = False
        with open(tex_input, 'r', encoding='utf8') as file:
            with open(tex_output, 'w', encoding='utf8') as out_file:
                for line in file:
                    if "\\title{tmp" in line:
                        line = line.replace('{tmp.', '{')
                        out_file.write(line)
                        line = "    \\author{Daniel W Rapp}\n"
                    elif "\\usepackage{amsmath}" in line:
                        out_file.write("    \\usepackage{amsmath}\n")
                        out_file.write("    \\usepackage{amsfonts}\n")
                        line = "    \\usepackage[mathscr]{euscript}\n"
                    elif "    % Colors for the hyperref package" in line:
                        out_file.write("    \\usepackage{float}\n")
                    elif "\\includegraphics" in line:
                        line = line.replace(",height=\\textheight", "")
                    elif "\\begin{figure}" in line:
                        line0 = line
                        line1 = next(file)
                        line2 = next(file).replace(",height=\\textheight", "")
                        if "pedigree" not in line2:
                            # we want to float the pedigree figure beacuse it is so large
                            line0 = line.replace("figure}", "figure}[H]")
                        out_file.write(line0)
                        out_file.write(line1)
                        line = line2
                    elif "We will generate all images so they have a width" in line:
                        skip = True
                    elif "\\captionsetup{labelformat=nolabel}" in line:
                        line = "\n"
                        skip = False
                    elif "\\begin{Verbatim}" in line:
                        out_file.write("\\scriptsize\n")
                    elif "\\end{Verbatim}" in line:
                        out_file.write(line)
                        line = "\\normalsize"
                    if not skip:
                        out_file.write(line)

        if not args.preserve:
            os.remove(tex_input)
    if not args.preserve:
        os.remove(out_file_name)
    sys.exit(rc)