# Execute this script from the /tasks/reacher_continuous_control directory
# bash ./setup_linux.sh

mkdir -p environments

# Download the reacher environment
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip --no-check-certificate

unzip Tennis_Linux.zip && mv Tennis_Linux environments/ && rm Tennis_Linux.zip
