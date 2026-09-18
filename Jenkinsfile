pipeline {
    agent {
        kubernetes {
            defaultContainer 'python'
            yaml '''
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
  containers:
    - name: jnlp
      image: m.daocloud.io/docker.io/jenkins/inbound-agent:3384.v60d89463d9e0-2-jdk25
      imagePullPolicy: IfNotPresent
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
    - name: python
      image: docker.io/library/python:3.11-slim
      imagePullPolicy: IfNotPresent
      command: ["sh", "-c"]
      args: ["cat"]
      tty: true
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
    - name: buildctl
      image: docker.io/moby/buildkit:v0.33.0-rootless
      imagePullPolicy: IfNotPresent
      command: ["sh", "-c"]
      args: ["cat"]
      tty: true
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
'''
        }
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
        skipDefaultCheckout(true)
    }

    triggers {
        pollSCM('H/10 * * * *')
    }

    parameters {
        string(
            name: 'REGISTRY_ENDPOINT',
            defaultValue: '10.43.250.50',
            description: '平台 Harbor Registry 地址'
        )
        string(
            name: 'REGISTRY_PROJECT',
            defaultValue: 'black-tea-simple',
            description: 'Harbor Project 名称'
        )
        string(
            name: 'IMAGE_NAME',
            defaultValue: 'black-tea-simple-data',
            description: 'Harbor 镜像仓库名称'
        )
    }

    environment {
        BUILDKIT_HOST = 'tcp://buildkit.platform-build.svc.cluster.local:1234'
        HARBOR_CREDENTIALS_ID = 'harbor-black-tea-simple-push'
        PYTORCH_UPSTREAM_IMAGE = 'm.daocloud.io/docker.io/pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime'
    }

    stages {
        stage('Checkout') {
            steps {
                script {
                    def scmVars = checkout scm
                    env.GIT_SHORT_SHA = scmVars.GIT_COMMIT.substring(0, 12)
                    env.IMAGE_TAG = "git-${env.GIT_SHORT_SHA}"
                    env.IMAGE_REF = "${params.REGISTRY_ENDPOINT}/${params.REGISTRY_PROJECT}/${params.IMAGE_NAME}:${env.IMAGE_TAG}"
                    env.PYTORCH_BASE_REF = "${params.REGISTRY_ENDPOINT}/${params.REGISTRY_PROJECT}/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
                }
                echo "Git revision: ${env.GIT_SHORT_SHA}"
                echo "Target image: ${env.IMAGE_REF}"
                echo "PyTorch base mirror: ${env.PYTORCH_BASE_REF}"
            }
        }

        stage('Static checks') {
            steps {
                sh '''
                    set -eu
                    python --version
                    python -m compileall -q src tools
                    test -s Dockerfile
                    test -s requirements.txt
                '''
            }
        }

        stage('Build and push image') {
            steps {
                container('buildctl') {
                    withCredentials([
                        usernamePassword(
                            credentialsId: env.HARBOR_CREDENTIALS_ID,
                            usernameVariable: 'HARBOR_USERNAME',
                            passwordVariable: 'HARBOR_PASSWORD'
                        )
                    ]) {
                        sh '''
                            set -eu
                            # Jenkins launches shell steps with tracing enabled. Disable it
                            # before deriving the registry auth token so the base64 value is
                            # never written to the console log.
                            set +x
                            export DOCKER_CONFIG="${WORKSPACE}/.docker"
                            trap 'rm -rf "${DOCKER_CONFIG}"' EXIT
                            mkdir -p "${DOCKER_CONFIG}"
                            AUTH="$(printf '%s:%s' "${HARBOR_USERNAME}" "${HARBOR_PASSWORD}" | base64 | tr -d '\n')"
                            printf '{"auths":{"%s":{"auth":"%s"}}}\n' \
                                "${REGISTRY_ENDPOINT}" "${AUTH}" \
                                > "${DOCKER_CONFIG}/config.json"
                            chmod 600 "${DOCKER_CONFIG}/config.json"

                            echo "Mirroring PyTorch base image into Harbor..."
                            buildctl \
                                --addr "${BUILDKIT_HOST}" \
                                build \
                                --frontend dockerfile.v0 \
                                --local context=. \
                                --local dockerfile=. \
                                --opt filename=Dockerfile.base \
                                --opt "build-arg:UPSTREAM_IMAGE=${PYTORCH_UPSTREAM_IMAGE}" \
                                --output "type=image,name=${PYTORCH_BASE_REF},push=true"

                            echo "Building application image..."
                            buildctl \
                                --addr "${BUILDKIT_HOST}" \
                                build \
                                --frontend dockerfile.v0 \
                                --local context=. \
                                --local dockerfile=. \
                                --opt filename=Dockerfile \
                                --opt "build-arg:PYTORCH_IMAGE=${PYTORCH_BASE_REF}" \
                                --output "type=image,name=${IMAGE_REF},push=true"
                        '''
                    }
                }
            }
        }
    }

    post {
        success {
            echo "镜像已推送：${env.IMAGE_REF}"
            echo "下一步将 deploy/overlays/dev/kustomization.yaml 的 newTag 更新为 ${env.IMAGE_TAG}，再由 Argo CD 同步。"
        }
    }
}
