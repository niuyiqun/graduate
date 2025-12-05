# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：test.py
@Author  ：niu
@Date    ：2025/12/5 13:36 
@Desc    ：
"""

import pickle
import uuid
from abc import abstractmethod, ABC
from typing import Optional, List
from datetime import datetime
from rank_bm25 import BM25Okapi

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
print(1)