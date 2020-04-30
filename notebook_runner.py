# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-30 13:40:16
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-30 13:53:09

import contextlib
import io
import logging
import sys

import nbformat
from IPython.core.formatters import format_display_data
from IPython.terminal.interactiveshell import InteractiveShell


''' Wrapper module around nbconvert allowing to run a notebook and redictect
    print/logging statements to the command line.

    Courtesy of Matthew Wardrop:
    https://gist.github.com/matthewwardrop/fe2148923048baabe14edacb2eda0b74
'''


class TeeOutput:

    def __init__(self, *orig_files):
        self.captured = io.StringIO()
        self.orig_files = orig_files

    def __getattr__(self, attr):
        return getattr(self.captured, attr)

    def write(self, data):
        self.captured.write(data)
        for f in self.orig_files:
            f.write(data)

    def get_output(self):
        self.captured.seek(0)
        return self.captured.read()


@contextlib.contextmanager
def redirect_logging(fh):
    old_fh = {}
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            old_fh[id(handler)] = handler.stream
            handler.stream = fh
    yield
    for handler in logging.getLogger().handlers:
        if id(handler) in old_fh:
            handler.stream = old_fh[id(handler)]


class NotebookRunner:

    def __init__(self, namespace=None):
        self.shell = InteractiveShell(user_ns=namespace)

    @property
    def user_ns(self):
        return self.shell.user_ns

    def run(self, nb, as_version=None, output=None, stop_on_error=True):
        if isinstance(nb, nbformat.NotebookNode):
            nb = nb.copy()
        elif isinstance(nb, str):
            nb = nbformat.read(nb, as_version=as_version)
        else:
            raise ValueError(f"Unknown notebook reference: `{nb}`")

        # Clean notebook
        for cell in nb.cells:
            cell.execution_count = None
            cell.outputs = []

        # Run all notebook cells
        for cell in nb.cells:
            if not self._run_cell(cell) and stop_on_error:
                break

        # Output the notebook if request
        if output is not None:
            nbformat.write(nb, output)

        return nb

    def _run_cell(self, cell):
        if cell.cell_type != 'code':
            return cell

        cell.outputs = []

        # Actually run the cell code
        stdout = TeeOutput(sys.stdout)
        stderr = TeeOutput(sys.stderr)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), redirect_logging(stderr):
            result = self.shell.run_cell(cell.source, store_history=True)

        # Record the execution count on the cell
        cell.execution_count = result.execution_count

        # Include stdout and stderr streams
        for stream, captured in {
            'stdout': self._strip_stdout(cell, stdout.get_output()),
            'stderr': stderr.get_output()
        }.items():
            if stream == 'stdout':
                captured = self._strip_stdout(cell, captured)
            if captured:
                cell.outputs.append(nbformat.v4.new_output('stream', name=stream, text=captured))

        # Include execution results
        if result.result is not None:
            cell.outputs.append(nbformat.v4.new_output(
                'execute_result', execution_count=result.execution_count, data=format_display_data(result.result)[0]
            ))
        elif result.error_in_exec:
            cell.outputs.append(nbformat.v4.new_output(
                'error',
                ename=result.error_in_exec.__class__.__name__,
                evalue=result.error_in_exec.args[0],
                traceback=self._render_traceback(
                    result.error_in_exec.__class__.__name__,
                    result.error_in_exec.args[0],
                    sys.last_traceback
                )
            ))

        return result.error_in_exec is None

    def _strip_stdout(self, cell, stdout):
        if stdout is None:
            return
        idx = max(
            stdout.find(f'Out[{cell.execution_count}]: '),
            stdout.find("---------------------------------------------------------------------------")
        )
        if idx > 0:
            stdout = stdout[:idx]
        return stdout

    def _render_traceback(self, etype, value, tb):
        """
        This method is lifted from `InteractiveShell.showtraceback`, extracting only
        the functionality needed by this runner.
        """
        try:
            stb = value._render_traceback_()
        except Exception:
            stb = self.shell.InteractiveTB.structured_traceback(etype, value, tb, tb_offset=None)
        return stb


def runNotebook(fname, save=False):
    ''' Open and run a notebook, and save "in-place" if specified. '''

    # Load notebook file without conversion
    with open(fname) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    # Execute notebook
    NotebookRunner().run(nb)

    # Save notebook in-place if specified
    if save:
        with open(fname, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
