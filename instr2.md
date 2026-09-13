### [x] gen_book_chapter.py msml610/01.2 --mode typst_aima --llm_backend hllm_cli

Do not print 

10:37:26 - INFO  _client.py _send_single_request:1025                   HTTP Request: GET https://openrouter.ai/api/v1/models "HTTP/1.1 200 OK"
10:37:28 - INFO  _client.py _send_single_request:1025                   HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
10:37:31 - INFO  _client.py _send_single_request:1025                   HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
10:37:34 - INFO  _client.py _send_single_request:1025                   HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
10:37:36 - INFO  _client.py _send_single_request:1025                   HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
10:37:38 - INFO  _client.py _send_single_request:1025                   HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
10:37:42 - INFO  _client.py _send_single_request:1025                   HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Is there a function to shut up the modules?
Print a tqdm progress bar instead

### [ ] Print the LLM cost and the conversion time, when available

### [x] Convert render_typst.sh into run_typst.py similar to

helpers_root/dev_scripts_helpers/documentation/run_latex.py
and
helpers_root/dev_scripts_helpers/documentation/notes_to_pdf.py

render_images.py is an option step

Assert if there are warnings, unless --no_abort_on_warnings

### [.] Understand why there are artifacts

render_typst.sh msml610/book/Lesson01.2-AI_and_Machine_Learning

generates

warning: no text within stars
    ┌─ msml610/book/Lesson01.2-AI_and_Machine_Learning.typ:145:2
    │
145 │   **determine how humans think**. Understanding human cognition is a complex
    │   ^^
    │
    = hint: using multiple consecutive stars (e.g. **) has no additional effect

warning: no text within stars
    ┌─ msml610/book/Lesson01.2-AI_and_Machine_Learning.typ:145:30
    │
145 │   **determine how humans think**. Understanding human cognition is a complex
    │                               ^^
    │
    = hint: using multiple consecutive stars (e.g. **) has no additional effect

is it a problem in the prompt?
Does it need a step of post-processing?
