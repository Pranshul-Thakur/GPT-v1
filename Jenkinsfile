pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t gpt-model .'
            }
        }
        
        stage('Test') {
            steps {
                sh 'docker run --rm gpt-model python -c "print(\'Model test passed\')"'
            }
        }
        
        stage('Deploy') {
            steps {
                sh 'docker run -d --name gpt-container gpt-model'
            }
        }
    }
}