---
language:
- en
pretty_name: "{dataset_name}"
tags:
- quickb
- text-chunking{gen_tag}
- {size_category}
task_categories:
- text-generation
- text-retrieval
task_ids:
- document-retrieval 
library_name: quickb
---

# {dataset_name}

Generated using [QuicKB](https://github.com/AdamLucek/quickb), a text chunking and synthetic question generation tool developed by [Adam Lucek](https://huggingface.co/AdamLucek).

## Dataset Details

### Chunking Configuration
- **Chunker**: {chunker_name}
- **Parameters**:
    {chunker_params}

{question_generation}

### Dataset Structure
This dataset contains {config_count} configurations:
1. `knowledgebase`: Contains the chunked text documents
   - Fields: id (string), text (string), source (string)
{train_config}