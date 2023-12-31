# Use Ubuntu as the base image
FROM ubuntu

# Set the working directory inside the container
WORKDIR /home/project_docker/

# Install required packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install pandas numpy seaborn matplotlib scikit-learn scipy geopandas

# Create a directory inside the container
RUN mkdir /home/project_docker/service-result/
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

# Copy the dataset to the container
COPY data.csv /home/project_docker/
COPY load.py /home/project_docker/
COPY dpre.py /home/project_docker/
COPY eda.py /home/project_docker/
COPY model.py /home/project_docker/
COPY vis.py /home/project_docker/


# Open the bash shell upon container startup
CMD ["/bin/bash"]