rm -f assignment4.zip
zip -r assignment4.zip . -x "*.git*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc"
