# HW#2 Frozen Lake

Repository of the code files of the assignment.
<ul>
  <li>"380-hw2-examplecode.ipynb" -- Start-up Notebook file (also its html file)</li>
  <li>"main.py" -- script file to run a policy in graphic mode</li>
</ul>

## Usage to run main.py

First create a virtual environment and install the dependencies:

```
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate for Windows PC
pip install -r requirements.txt
```
then
```
- 'python main.py' -- episodes will run using a random fixed policy, or
- 'python main.py policy.txt' -- episodes will run using a give policy stored
                                   in a file, where a policy is written in one
                                   line separated by spaces, e.g.
                                   '3 1 0 3 2 ...'
```


## Acknowledgments

Maintained by Noriko Tomuro
