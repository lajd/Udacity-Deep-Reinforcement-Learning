# Execute this script from the /tasks/reacher directory
# bash ./setup_linux.sh

mkdir -p environments

# Download the reacher environment
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip --no-check-certificate

unzip Reacher_Linux.zip && mv Reacher_Linux environments/ && rm Reacher_Linux.zip
