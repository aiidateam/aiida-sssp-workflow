# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, redefined-outer-name
import pytest


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):  # pylint: disable=unused-argument
    return "docker-compose.aiida-sssp-workflow.yml"


@pytest.fixture(scope="session")
def docker_compose(docker_services):
    # pylint: disable=protected-access
    return docker_services._docker_compose


def is_container_ready(docker_compose):
    output = (
        docker_compose.execute("exec -T aiida verdi status || true").decode().strip()
    )
    return "Connected to RabbitMQ" in output and "Daemon is running" in output


@pytest.fixture(scope="session", autouse=True)
def _docker_service_wait(docker_services):
    """Container startup wait."""
    docker_compose = docker_services._docker_compose

    docker_services.wait_until_responsive(
        timeout=120.0, pause=0.1, check=lambda: is_container_ready(docker_compose)
    )


@pytest.fixture
def container_user():
    return "aiida"


@pytest.fixture
def aiida_exec(docker_compose):
    def execute(command, user=None, **kwargs):
        if user:
            command = f"exec -T --user={user} aiida {command}"
        else:
            command = f"exec -T aiida {command}"
        return docker_compose.execute(command, **kwargs)

    return execute
