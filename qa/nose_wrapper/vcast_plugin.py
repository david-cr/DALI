"""Nose 1.x plugin that dumps VectorCAST per-test coverage data.

Loaded automatically by qa/nose_wrapper/__main__.py via addplugins=[...] so
every nose 1.x test run gets the same coverage hook as nose2/unittest tests
get via teardown_function / tearDown. No per-file or per-test code needed.
"""
from nose.plugins import Plugin


class VCastCoverage(Plugin):
    name = "vcast-coverage"
    enabled = True

    def options(self, parser, env):
        # Always enabled; --with-vcast-coverage is optional and harmless.
        super().options(parser, env)

    def configure(self, options, conf):
        super().configure(options, conf)
        self.enabled = True

    def afterTest(self, test):
        """Fires after every test (function, class method, generator, parametrized)."""
        try:
            name_tag = test.id().split(".")[-1]
        except Exception:
            name_tag = "unknown_test"
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
            # vcast modules not present (e.g. non-instrumented build) - do not
            # break test runs in that case.
            pass
