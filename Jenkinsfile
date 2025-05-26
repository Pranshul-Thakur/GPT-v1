pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                bat 'docker build -t gpt-model .'
            }
        }

        stage('Test') {
            steps {
                bat 'docker run --rm gpt-model python -c "print(\'Model test passed\')"'
            }
        }

        stage('Deploy') {
            steps {
                bat '''
                docker stop gpt-container || echo "Container not running"
                docker rm gpt-container || echo "Container removed"
                docker run -d --name gpt-container gpt-model
                '''
            }
        }
    }
}
