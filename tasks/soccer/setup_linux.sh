# Execute this script from the /tasks/reacher_continuous_control directory
# bash ./setup_linux.sh

mkdir -p environments

# Download the reacher environment
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip --no-check-certificate

unzip Soccer_Linux.zip && mv Soccer_Linux environments/ && rm Soccer_Linux.zip


