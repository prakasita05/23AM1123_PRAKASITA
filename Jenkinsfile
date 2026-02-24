pipeline {
    agent any

    stages {

        stage('GitHub Access') {
            steps {
                echo 'Repository Cloned Successfully'
            }
        }

        stage('Environment Setup') {
            steps {
                dir('exp 2') {
                    bat 'python --version'
                    bat 'python -m ensurepip --default-pip'
                    bat 'python -m pip install --upgrade pip'
                    bat 'python -m pip install -r requirement.txt'
                }
            }
        }

        stage('Training Stage') {
            steps {
                dir('exp 2') {
                    echo 'Starting Model Training...'
                    bat 'python train.py'
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
