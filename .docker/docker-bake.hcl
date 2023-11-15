# docker-bake.hcl
variable "ORGANIZATION" {
  default = "unkcpz"
}

variable "REGISTRY" {
  default = "docker.io/"
}

variable "PLATFORMS" {
  default = ["linux/amd64"]
}

variable "QE_VERSION" {
  default = "7.2"
}

function "tags" {
  params = [image]
  result = [
    "${REGISTRY}${ORGANIZATION}/${image}:newly-baked"
  ]
}

group "default" {
  targets = ["aiida-sssp-workflow"]
}

target "aiida-sssp-workflow" {
  tags = tags("aiida-sssp-workflow")
  contexts = {
    src = ".."
  }
  platforms = "${PLATFORMS}"
  args = {
    "QE_VERSION" = "${QE_VERSION}"
  }
}
