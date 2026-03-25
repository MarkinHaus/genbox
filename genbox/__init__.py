"""
genbox — local AI generation toolkit
usage: from genbox import pipeline, models, config
"""
from genbox.config import cfg
from genbox import models
# pipeline imported lazily to avoid heavy ML imports at CLI startup
