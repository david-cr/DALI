"""Nose2 plugin that dumps VectorCAST coverage at end of run.

Calling the VCAST coverage dump after each test caused cudaErrorIllegalAddress
on subsequent tests (the dump interacts badly with DALI's asynchronous GPU
work-cleanup path).  We defer the dump to stopTestRun so it executes once after
all tests have finished, by which point DALI's async work is fully drained.

This loses per-test attribution but keeps the coverage data and the tests
crash-free.
"""
from nose2.events import Plugin


class VCastCoverage(Plugin):
    configSection = "vcast-coverage"
    commandLineSwitch = (None, "with-vcast-coverage", "Dump VectorCAST coverage at end of run")
    alwaysOn = True

    def stopTestRun(self, event):
        """Fires once at the end of the test run, after all tests complete."""
        name_tag = "test_run"
        try:
            import dali_core_vcast
            import dali_kernel_vcast
            import dali_operator_vcast
            import dali_pipeline_vcast

            dali_core_vcast.dali_core_dump_vcast_data(name_tag)
            dali_kernel_vcast.dali_kernel_dump_vcast_data(name_tag)
            dali_operator_vcast.dali_operator_dump_vcast_data(name_tag)
            dali_pipeline_vcast.dali_pipeline_dump_vcast_data(name_tag)
        except ImportError:
            # vcast modules not present in non-instrumented builds; do not break tests.
            pass
