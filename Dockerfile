# A helpful blog on Docker for a Python app:
# https://www.docker.com/blog/containerized-python-development-part-1/

FROM python:3.6-slim
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

# the following RUN command was derived from the link below
# these are meant to reduce the image size
# pandas, scipy, dash & dash_bootstrap_components occupy
# 400MB with default pip install options
# 80MB are saved with --no-cache-dir
# https://towardsdatascience.com/how-to-shrink-numpy-scipy-pandas-and-matplotlib-for-your-data-product-4ec8d7e86ee4
COPY requirements_aws_ec2.txt /docker_app/
RUN pip install --no-cache-dir -r /docker_app/requirements_aws_ec2.txt

# copy app source and data files
COPY . /docker_app/

WORKDIR /docker_app/

# expose Dash Flask port
EXPOSE 8050

# run the app
CMD ["python", "app.py"]