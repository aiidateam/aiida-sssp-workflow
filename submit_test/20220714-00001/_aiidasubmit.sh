#!/bin/bash
exec > _scheduler-stdout.txt
exec 2> _scheduler-stderr.txt


'/home/aiida/.conda/env/quantum-espresso-6.8/bin/ld1.x' < 'aiida.in' > 'aiida.out' 
