# Memory Augmented Multimodal Research


This reading list is composed of three comprehensive parts:

1. **Multimodal Context Modeling with Memory**: This section explores how memory mechanisms are employed to model and retain long-term contextual information across various modalities, such as audio and video. It covers techniques for enhancing the understanding and processing of continuous and complex multimodal data.

2. **Multimodal Contextual Memory for Vision and Robotics**: This part focuses on the integration of memory mechanisms to augment the capabilities of embodied agents and robotics. It includes applications such as long-term planning, decision-making, visual navigation, and manipulation reasoning, demonstrating how memory can significantly enhance an agent's ability to operate in complex environments.

3. **External Multimodal Knowledge/Memory Augmentation**: Here, we delve into the use of memory as an external knowledge source. This external information serves as complementary data that can enhance the performance and capabilities of models by providing additional context and insights, thereby improving their overall effectiveness.




Table of Contents

- [1 Multimodal Context Modeling with Memory](#1-multimodal-context-modeling-with-memory)
    - [1.1 Audio Context Modeling with Memory](#11-audio-context-modeling-with-memory)
    - [1.2 Video Context Modeling with Memory](#12-video-context-modeling-with-memory)
        - [1.2.1 Video Representation](#121-video-representation)
        - [1.2.2 Large Video Language Models](#122-large-video-language-models)
        - [1.2.3 Video Agent](#123-video-agent)
        - [1.2.4 Video Caption](#124-video-caption)
        - [1.2.5 Video QA](#125-videoqa)
        - [1.2.6 Video Summarization](#126-video-summarization)
        - [1.2.7 Action Claasification / Localization](#127-action-classification--localization)
        - [1.2.8 Video Object Segmentation / Tracking](#128-video-object-segmentation--tracking)
        - [1.2.9 Video Generation](#129-video-generation)
        - [1.2.10 Other Tasks](#1210-other-tasks)
    - [1.3 Other Modality Context Modeling with Memory](#13-other-modality-context-modeling-with-memory)
- [2 Multimodal Contextual Memory for Vision and Robotics](#2-multimodal-contextual-memory-for-vision-and-robotics)
    - [2.1 Visual Navigation](#21-visual-navigation)
    - [2.2 Visual Odometry](#22-visual-odometry)
    - [2.3 Manipulation Reasoning \& Planning](#23-manipulation-reasoning--planning)
    - [2.4 Multimodal Memory Augmented Agent](#24-multimodal-memory-augmentated-agent)
- [3 External Multimodal Knowledge/Memory Augmentation](#3-external-multimodal-knowledgememory-augmentation)
    - [3.1 External Audio/Speech Knowledge/Memory](#31-external-audiospeech-knowledgememory)
        - [3.1.1 Audio Caption](#311-audio-caption)
        - [3.1.2 ASR](#312-asr)
        - [3.1.3 Text-to-Audio](#313-text-to-audio)
    - [3.2 External Image Knowledge/Memory](#32-external-image-knowledgememory)
        - [3.2.1 Image Classification](#321-image-classification)
        - [3.2.2 Image Segmentation](#322-image-segmentation)
        - [3.2.3 Image Retrieval](#323-image-retrieval)
        - [3.2.4 Image Caption](#324-image-caption)
        - [3.2.5 VQA](#325-vqa)
        - [3.2.6 Visual Dialog](#326-visual-dialogue)
        - [3.2.7 Machine Translation](#327-machine-translation)
        - [3.2.8 Visual Representation](#328-visual-representation)
        - [3.2.9 Image Generation](#329-image-generation)
    - [3.3 External Video Knowledge/Memory](#33-external-video-knowledgememory)
        - [3.3.1 Video Caption](#331-video-caption)
        - [3.3.2 VideoQA](#332-videoqa)
    - [3.4 External 3D Knowledge/Memory](#34-external-3d-knowledgememory)
    - [3.5 Retrieval-Augmented Multimodal Large Language Model](#35-retrieval-augmented-multimodal-large-language-model)
    - [3.6 Retrieval-Augmented Multimodal Agent](#36-retrieval-augmented-multimodal-agent)






## 1 Multimodal Context Modeling with Memory


### 1.1 Audio Context Modeling with Memory

- Memory-augmented conformer for improved end-to-end long-form ASR `Interspeech 2023`
- Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing `Arxiv 2023-10`
- MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage `Arxiv 2024-03`


### 1.2 Video Context Modeling with Memory

#### 1.2.1 Video Representation
#transformer
- Memory Consolidation Enables Long-Context Video Understanding `CVPR 2024`


#### 1.2.2 Large Video Language Models

- MovieChat: From Dense Token to Sparse Memory for Long Video Understanding `CVPR 2024`
- MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding `CVPR 2024`
- Streaming Long Video Understanding with Large Language Models `Arxiv 2024-05`
- Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams `Arxiv 2024-06`

#vertical category
- OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning `Arxiv 2024-05`

#### 1.2.3 Video Agent

- ChatVideo: A Tracklet-centric Multimodal and Versatile Video Understanding System `Arxiv 2023-04`
- A Simple LLM Framework for Long-Range Video Question-Answering `Arxiv 2023-12`
- LifelongMemory: Leveraging LLMs for Answering Queries in Egocentric Videos `Arxiv 2023-12`
- DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models `Arxiv 2024-01`
- Language Repository for Long Video Understanding `Arxiv 2024-03`
- VideoAgent: Long-form Video Understanding with Large Language Model as Agent `Arxiv 2403`
- VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding `ECCV2024`
- DrVideo: Document Retrieval Based Long Video Understanding `Arxiv 2024-06`



#### 1.2.4 Video Caption
#RNN;LSTM;GRU
- Multimodal Memory Modelling for Video Captioning `CVPR 2018`
- Memory-Attended Recurrent Network for Video Captioning `CVPR 2019`

#transformer
- MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning `ACL 2020`
- AAP-MIT: Attentive Atrous Pyramid Network and Memory Incorporated Transformer for Multisentence Video Description `TIP 2022`


#### 1.2.5 VideoQA

#RNN;LSTM;GRU
- Heterogeneous Memory Enhanced Multimodal Attention Model for Video Question Answering `CVPR 2019`
- Feature Augmented Memory with Global Attention Network for VideoQA `IJCAI 2020`

#memory network
- A Read-Write Memory Network for Movie Story Understanding `CVPR 2017`

#transformer
- Glance and Focus: Memory Prompting for Multi-Event Video Question Answering `NeurIPS 2023`


#### 1.2.6 Video Summarization
#LSTM
- Video Summarization with Long Short-term Memory `ECCV 2016`
- Stacked Memory Network for Video Summarization `MM 2019`

#memory network
- Extractive Video Summarizer with Memory Augmented Neural Networks `MM 2018`

#### 1.2.7 Action Classification \& Localization

#memory network
- Memory-Augmented Temporal Dynamic Learning for Action Recognition `AAAI 2019`

#transformer
- Long Short-Term Transformer for Online Action Detection `NeurIPS 2021`
- MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition `CVPR 2022`
- Recurring the Transformer for Video Action Recognition `CVPR 2022`
- Memory-and-Anticipation Transformer for Online Action Understanding `ICCV 2023`
- Online Temporal Action Localization with Memory-Augmented Transformer `ECCV 2024`


#### 1.2.8 Video Object Segmentation \& Tracking

#RNN;LSTM;GRU
- Learning Video Object Segmentation with Visual Memory `ICCV 2017`
- Recurrent Filter Learning for Visual Tracking `ICCV 2017 Workshop`
- Learning Recurrent Memory Activation Networks for Visual Tracking `TIP 2020`

#memory network
- Learning Dynamic Memory Networks for Object Tracking `ECCV 2018`
- Video Object Segmentation using Space-Time Memory Networks `ICCV 2019`
- Rethinking space-time networks with improved memory coverage for efficient video object segmentation `NeurIPS 2021`
- STMTrack: Template-free Visual Tracking with Space-time Memory Networks `CVPR 2021`
- Motion-aware Memory Network for Fast Video Salient Object Detection `TIP 2022`
- Robust and Efficient Memory Network for Video Object Segmentation `ICME 2023`


#Graph
- Dual Temporal Memory Network for Efficient Video Object Segmentation `MM 2020`
- Video Object Segmentation with Episodic Graph Memory Networks `ECCV 2020`

#transformer
- Associating Objects with Transformers for Video Object Segmentation `NeurIPS 2021`
- Local Memory Attention for Fast Video Semantic Segmentation `IROS 2021`
- TrackFormer: Multi-Object Tracking with Transformers `CVPR 2022`
- XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model `ECCV 2022`
- Streaming Video Model `CVPR 2023`
- Multiscale Memory Comparator Transformer for Few-Shot Video Segmentation `Arxiv 2023-07`
- Reading Relevant Feature from Global Representation Memory for Visual Object Tracking `NeurIPS 2023`
- RMem: Restricted Memory Banks Improve Video Object Segmentation `CVPR 2024`
- Efficient Video Object Segmentation via Modulated Cross-Attention Memory `Arxiv 2024-03`
- SAM 2: Segment Anything in Images and Videos `Arxiv 2024-08`

#others
- Alignment Before Aggregation: Trajectory Memory Retrieval Network for Video Object Segmentation `ICCV 2023`

#### 1.2.9 Video Generation
#RNN;LSTM;GRU
- Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning `CVPR 2021`

#memory network
- MV-TON: Memory-based Video Virtual Try-on network `MM 2021`
- Audio-driven Talking Face Video Generation with Learning-based Personalized Head Pose `TMM 2022`
- SyncTalkFace: Talking Face Generation with Precise Lip-Syncing via Audio-Lip Memory `AAAI 2022`
- EMMN: Emotional Motion Memory Network for Audio-driven Emotional Talking Face Generation `ICCV 2023`

#3d-cnn
- STAM: A SpatioTemporal Attention Based Memory for Video Prediction `TMM 2022`

#transformer
- Memories are One-to-Many Mapping Alleviators in Talking Face Generation `PAMI 2022`


#### 1.2.10 Other Tasks

##### 1.2.10.1 Flow Estimation
#transformer
- MemFlow: Optical Flow Estimation and Prediction with Memory `CVPR 2024`

##### 1.2.10.2 Depth Estimation
#transformer
- MAMo: Leveraging Memory and Attention for Monocular Video Depth Estimation `ICCV 2023`

##### 1.2.10.3 Video Deblurring
#memory network
- Multi-Scale Memory-Based Video Deblurring `CVPR 2022`

##### 1.2.10.4 Gesture recognition
#memory network
- MENet: a Memory-Based Network with Dual-Branch for Efficient Event Stream Processing `ECCV 2022`

##### 1.2.10.5 visual speech recognition
#memory network
- Multi-modality Associative Bridging through Memory: Speech Sound Recollected from Face Video `ICCV 2021`
- Distinguishing Homophenes Using Multi-Head Visual-Audio Memory for Lip Reading `AAAI 2022`
- CroMM-VSR: Cross-Modal Memory Augmented Visual Speech Recognition `TMM 2022`



### 1.3 Other Modality Context Modeling with Memory

#### 1.3.1 Image Caption
#memory network
- Attend to You: Personalized Image Captioning with Context Sequence Memory Networks `CVPR 2017`

#### 1.3.2 Coloralization
#memory network
- Coloring With Limited Data: Few-Shot Colorization via Memory-Augmented Networks `CVPR 2019`

#### 1.3.3 Deraining
#memory network
- Memory Oriented Transfer Learning for Semi-Supervised Image Deraining `CVPR 2021`

#### 1.3.4 Semantic Segment
#memory network
- Memory-based Semantic Segmentation for Off-road Unstructured Natural Environments `IROS 2021`
- Learning Meta-class Memory for Few-Shot Semantic Segmentation `ICCV 2021`
- Remember the Difference: Cross-Domain Few-Shot Semantic Segmentation via Meta-Memory Transfer `CVPR 2022`

#### 1.3.5 Anomaly Detection
#memory network
- Divide-and-Assemble: Learning Block-wise Memory for Unsupervised Anomaly Detection `ICCV 2021`

#### 1.3.6 Face Super Resolution
#memory network
- Universal Face Restoration With Memorized Modulation `Arxiv 2021`

#### 1.3.7 Image to Image Translation
#memory network
- Memory-guided Unsupervised Image-to-image Translation `CVPR 2021`

### 1.3.8 3D Scene Reconstruction
#transformer
- TransformerFusion: Monocular {RGB} Scene Reconstruction using Transformers `NeurIPS 2021`







## 2 Multimodal Contextual Memory for Vision and Robotics

### 2.1 Visual Navigation

#RNN#LSTM#GRU
- Visual Memory for Robust Path Following `NeurIPS 2018`
- Structured Scene Memory for Vision-Language Navigation `CVPR 2021`

#graph
- Visual Graph Memory with Unsupervised Representation for Visual Navigation `ICCV 2021`
- MemoNav: Working Memory Model for Visual Navigation `Arxiv 2024-02`

#transformer
- Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks `CVPR 2019`
- Memory-Augmented Reinforcement Learning for Image-Goal Navigation `IROS 2022` 

#benchmark
- MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation `NeurIPS 2020`

### 2.2 Visual Odometry

#RNN#LSTM#GRU
- Deep Visual Odometry With Adaptive Memory `PAMI 2022`



### 2.3 Manipulation Reasoning \& Planning

#transformer
- Out of Sight, Still in Mind: Reasoning and Planning about Unobserved Objects with Video Tracking Enabled Memory Models `ICRA2024`


#### 2.4 Multimodal Memory Augmentated Agent

#transoformer
- GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation `Arxiv 2023-07`
- JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models `Arxiv 2023-11`
- Explore, select, derive, and recall: Augmenting llm with human-like memory for mobile task automation `Arxiv 2023-12`
- AppAgent: Multimodal Agents as Smartphone Users `Arxiv 2023-12`
- Multimodal Embodied Interactive Agent for Cafe Scene `Arxiv 2024-02`
- OS-Copilot: Towards Generalist Computer Agents with Self-Improvement `Arxiv 2024-02`





## 3 External Multimodal Knowledge/Memory Augmentation


### 3.1 External Audio/Speech Knowledge/Memory

#### 3.1.1 Audio Caption
- Audio Captioning using Pre-Trained Large-Scale Language Model Guided by Audio-based Similar Caption Retrieval `Arxiv 2012`
- Recap: Retrieval-Augmented Audio Captioning `ICASSP 2024`

#### 3.1.2 ASR
- Using External Off-Policy Speech-To-Text Mappings in Contextual End-To-End Automated Speech Recognition `Arxiv 2023-01`
- Retrieval Augmented End-to-End Spoken Dialog Models `ICASSP 2024`

#### 3.1.3 Text-To-Audio

- Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models `ICML 2023`
- Retrieval-Augmented Text-to-Audio Generation `ICASSP 2024`

### 3.2 External Image Knowledge/Memory

#### 3.2.1 Image Classification

- With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations `ICCV 2021`
- A Memory Transformer Network for Incremental Learning `BMVC 2022`
- Improving Image Recognition by Retrieving from Web-Scale Image-Text Data `CVPR 2023`
- Retrieval-Enhanced Visual Prompt Learning for Few-shot Classification `Arxiv 2023-06`
- Towards flexible perception with visual memory `Arxiv 2023-08`

#### 3.2.2 Image Segmentation

- kNN-CLIP: Retrieval Enables Training-Free Segmentation on Continually Expanding Large Vocabularies `Arxiv 2024-04`

#### 3.2.3 Image Retrieval

- Knowledge-Enhanced Dual-stream Zero-shot Composed Image Retrieval `CVPR 2024`

#### 3.2.4 Image Caption

#RNN;LSTM;GRU
- Recurrent Relational Memory Network for Unsupervised Image Captioning `IJCAI 2020`

#transformer
- Memory-Augmented Image Captioning `AAAI 2021`
- Retrieval-Augmented Transformer for Image Captioning `CBMI 2022`
- Smallcap: Lightweight Image Captioning Prompted with Retrieval Augmentation `CVPR 2023`
- AMA: Adaptive Memory Augmentation for Enhancing Image Captioning `BMVC 2023`
- Retrieval-augmented Image Captioning `EACL 2023`
- With a Little Help from your own Past: Prototypical Memory Networks for Image Captioning `ICCV 2023`
- Re-ViLM: Retrieval-Augmented Visual Language Model for Zero and Few-Shot Image Captioning `EMNLP 2023`
- EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension `CVPR 2024`
- MeaCap: Memory-Augmented Zero-shot Image Captioning `CVPR 2024`
- Understanding Retrieval Robustness for Retrieval-augmented Image Captioning `ACL 2024`

#### 3.2.5 VQA

- An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA `AAAI 2022`
- KAT: A Knowledge Augmented Transformer for Vision-and-Language `NAACL 2022`
- MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text `EMNLP 2022`
- Retrieval Augmented Visual Question Answering with Outside Knowledge `EMNLP 2022`
- Generate then Select: Open-ended Visual Question Answering Guided by World Knowledge `ACL 2023`
- Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering `NeurIPS 2023`
- Reveal: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory `CVPR 2023`
- Multimodal Prompt Retrieval for Generative Visual Question Answering `ACL 2023`
- GeReA: Question-Aware Prompt Captions for Knowledge-based Visual Question Answering `Arxiv 2024-02`

#### 3.2.6 Visual Dialogue

- Augmenting Transformers with KNN-Based Composite Memory for Dialog `TACL 2021`
- Maria: A Visual Experience Powered Conversational Agent `ACL 2021`
- Text is NOT Enough: Integrating Visual Impressions into Open-domain Dialogue Generation `MM 2021`

#### 3.2.7 Machine Translation

- Neural Machine Translation with Universal Visual Representation `ICLR 2020`
- Neural Machine Translation with Phrase-Level Universal Visual Representations `ACL 2022`

#### 3.2.8 Visual Representation


- K-LITE: Learning Transferable Visual Models with External Knowledge `NeurIPS 2022`
- Improving CLIP Training with Language Rewrites `NeurIPS 2023`
- Retrieval-Enhanced Contrastive Vision-Text Models `ICLR 2024`
- Learning Customized Visual Models with Retrieval-Augmented Knowledge `CVPR 2024`
- Understanding Retrieval-Augmented Task Adaptation for Vision-Language Models `ICML 2024`

#### 3.2.8 Image Enhancement

- Glow in the Dark: Low-Light Image Enhancement With External Memory `TMM 2024`

#### 3.2.9 Image Generation

- RetrieveGAN: Image Synthesis via Differentiable Patch Retrieval `ECCV 2020`
- Instance-Conditioned GAN `NeurIPS 2021`
- Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination `EMNLP 2022`
- Retrieval-Augmented Diffusion Models `NeurIPS 2022`
- Re-Imagen: Retrieval-Augmented Text-to-Image (diffusion) Generator `ICLR 2023`
- kNN-Diffusion: Image Generation via Large-Scale Retrieval `ICLR 2023`
- Retrieval-Augmented Multimodal Language Modeling `ICML 2023`

### 3.3 External Video Knowledge/Memory

#### 3.3.1 Video Caption

- Incorporating Background Knowledge into Video Description Generation `EMNLP 2018`
- Memory-Based Augmentation Network for Video Captioning `TMM 2024`
- Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval `CVPR 2024`
- Retrieval-Augmented Egocentric Video Captioning `Arxiv 2024-01`

#### 3.3.2 VideoQA

- Retrieving-to-Answer: Zero-Shot Video Question Answering with Frozen Large Language Models `Arxiv 2023-06`

### 3.4 External 3D Knowledge/Memory

- ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model `ICCV 2023`
- AMD: Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion `AAAI 2024`
- Retrieval-Augmented Score Distillation for Text-to-3D Generation `Arxiv 2024-02`

### 3.5 Retrieval-Augmented Multimodal Large Language Model

- RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model  `Arxiv 2024-01`
- Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs `Arxiv 2024-04`
- Reminding Multimodal Large Language Models of Object-aware Knowledge with Retrieved Tags `Arxiv 2024-06`

### 3.6 Retrieval-Augmented Multimodal Agent

- Avis: Autonomous visual information seeking with large language model agent `NeurIPS 2023`
- Mp5: A multi-modal open-ended embodied system in minecraft via active perception `CVPR 2024`
- SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge `Arxiv 2024-05`
- Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs `Arxiv 2024-05`



