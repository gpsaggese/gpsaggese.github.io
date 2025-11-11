- GitHub: https://github.com/gpsaggese/umd_data605
- Dockerhub: 


cd ~/src/cmsc424-fall2022/Assignment-0
docker build -t amolumd/cmsc424-fall2022 .

docker run --rm -ti -p 8888:8888 -p 8881:8881 -p 5432:5432 -v $(pwd):/data amolumd/cmsc424-fall2022

docker> jupyter-notebook --port=8888 --allow-root --no-browser --ip=0.0.0.0


# Old container
> docker run -it --rm -v $(pwd):/data --name my-postgres-container kostasxirog/cmsc642-postgresql bash

# To build Docker image

```
> docker build --tag umd_data605 .
```


# Postgres

more /var/log/postgresql/postgresql-14-main.log

https://stackoverflow.com/questions/38466190/cant-connect-to-postgresql-on-port-5432

vi /etc/postgresql/14/main/postgresql.conf

service postgresql restart

# From outside

> psql -U postgres -h localhost
psql (14.5 (Homebrew))
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
Type "help" for help.

postgres=#
