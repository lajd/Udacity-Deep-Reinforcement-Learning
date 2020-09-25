# Execute this script from the /tasks/crawler directory
# bash ./setup_linux.sh

mkdir -p environments

# Download the crawler environment
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip --no-check-certificate

unzip Crawler_Linux.zip && mv Crawler_Linux environments/ && rm Crawler_Linux.zip
