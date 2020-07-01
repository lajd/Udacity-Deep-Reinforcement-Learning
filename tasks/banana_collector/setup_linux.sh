# Execute this script from the /tasks/banana_collector directory
# bash ./setup_linux.sh

mkdir -p environments

# Download banana environment
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip --no-check-certificate
# Download VisualBanana environment
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip --no-check-certificate

unzip Banana_Linux.zip && mv Banana_Linux environments/ && rm Banana_Linux.zip
unzip VisualBanana_Linux.zip && mv VisualBanana_Linux environments/ && rm VisualBanana_Linux.zip
