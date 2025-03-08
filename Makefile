gce-instance:
	gcloud compute instances create trading-instance \
	--zone=asia-southeast1-c \
	--machine-type=n1-standard-2 \
	--accelerator=type=nvidia-tesla-t4,count=1 \
	--maintenance-policy=TERMINATE \
	--image-family=pytorch-latest-gpu \
	--image-project=deeplearning-platform-release \
	--boot-disk-size=200GB \
	--scopes=https://www.googleapis.com/auth/cloud-platform \
	--metadata="install-nvidia-driver=True"
