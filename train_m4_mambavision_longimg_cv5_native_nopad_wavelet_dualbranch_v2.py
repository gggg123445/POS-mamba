"""Standalone training entry point for POS-Mamba."""
from pos_mamba.models import (SpatialGatedFusion, HaarWaveletFreqBranchV2,
                              MambaVisionLongImage6TaskV2)
from pos_mamba.cli import main

if __name__ == '__main__':
    main()
