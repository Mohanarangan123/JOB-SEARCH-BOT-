"""
sources package — exports a convenience factory for the default SourceRegistry.
"""
from __future__ import annotations

from job_discovery.sources.base import SourceRegistry
from job_discovery.sources.cutshort import CutshortAdapter
from job_discovery.sources.generic import GenericAdapter
from job_discovery.sources.hirect import HirectAdapter
from job_discovery.sources.hirist import HiristAdapter
from job_discovery.sources.indeed import IndeedAdapter
from job_discovery.sources.instahyre import InstahyreAdapter
from job_discovery.sources.linkedin import LinkedInAdapter
from job_discovery.sources.naukri import NaukriAdapter
from job_discovery.sources.wellfound import WellfoundAdapter


def build_default_registry() -> SourceRegistry:
    """
    Create a SourceRegistry pre-loaded with all known adapters.
    GenericAdapter is always registered last (catches everything).
    """
    registry = SourceRegistry()
    # Tier 1
    registry.register(LinkedInAdapter())
    registry.register(IndeedAdapter())
    registry.register(NaukriAdapter())
    # Tier 2
    registry.register(CutshortAdapter())
    registry.register(InstahyreAdapter())
    registry.register(HiristAdapter())
    registry.register(WellfoundAdapter())
    registry.register(HirectAdapter())
    # Fallback
    registry.register(GenericAdapter())
    return registry
