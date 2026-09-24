# Copy to hpc/config.local.sh and edit. Paths must be visible on compute nodes.
export CONTAINER_RUNTIME=singularity
export PIXI_CACHE_DIR=/data/core-nrad-liebig/chemingw/INIM_NIPipeline
export CONTAINER_DIR=/data/core-nrad-liebig/chemingw/INIM_NIPipeline/containers
export SINGULARITY_CACHEDIR=/data/core-nrad-liebig/chemingw/INIM_NIPipeline/singularity-cache
export BIDS_DIR=/replace/with/BIDS
export FS_LICENSE_DIR=/replace/with/freesurfer-license-directory
# LIT uses code/weights from its image. Unset an earlier external override.
unset LIT_REPO
# Optional only for an external compatible LIT checkout WITH model weights:
# export LIT_REPO=/project/software/LIT
