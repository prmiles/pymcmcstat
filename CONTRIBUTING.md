# Guidelines for Contributing

Individual or groups are very welcome to contribute to pymcmcstat.  There are four main ways of contributing to the pymcmcstat project:

1. Adding new or improved functionality to the existing codebase.
2. Fixing outstanding issues with the existing codebase.
3. Contributing or improving the documentation (`doc`).
4. Submitting issues related to bugs or desired enhancements.

# Opening issues

The best way to announce issues is via [Github Issue Tracker](https://github.com/prmiles/pymcmcstat/issues).  Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for key words in the project issue tracker.

# Contributing code via pull requests

If you would like to address an issue directly, users are strongly encouraged to submit patches for new or existing issues via pull requests.  This is especially appreciated for simple fixes like typos or tweaks to documentation.

Contributors are also encouraged to contribute new code to enhance pymcmcstat's functionality, also via pull requests. Please consult the [pymcmcstat documentation](https://pymcmcstat.readthedocs.io/) to ensure that any new contribution does not strongly overlap with existing functionality.

The recommended workflow for contributing to pymcmcstat is to fork the [GitHub repository](https://github.com/prmiles/pymcmcstat), clone it to your local machine, and develop on a feature branch.

## Steps:

1. Fork the [project repository](https://github.com/prmiles/pymcmcstat) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the pymcmcstat repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/pymcmcstat.git
   $ cd pymcmcstat
   $ git remote add upstream git@github.com:prmiles/pymcmcstat.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never routinely work on the ``master`` branch of any repository.

4. Project requirements are in ``requirements.txt``. It is recommended that you use some type of [virtual environment](https://docs.python.org/3/tutorial/venv.html) for your code development.  To install the required packages run:

   ```bash
   $ pip install -r requirements.txt
   ```

5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the pymcmcstat repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

## Pull request checklist

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

*  All public methods must have informative docstrings with sample usage when appropriate.

*  Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

*  All other tests pass when everything is rebuilt from scratch.

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Run any of the pre-existing [tutorials](https://github.com/prmiles/notebooks/blob/master/pymcmcstat/index.ipynb) that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit test, but also to show how your contribution improves pymcmcstat for end users.

* *A local tutorials directory is currently being developed.*

You can also check for common programming errors with the following
tools:

* Check code **coverage** (at least 80%) with:

  ```bash
  $ pip install coveralls
  $ coverage run --source=pymcmcstat -m unittest test/<path to test>/test_<name>.py
  $ coverage report --fail-under=80
  ```
Note that Travis-CI will run
  ```bash
  - coverage run --source=pymcmcstat -m unittest discover -s "test" -p "test*.py"
  - coverage report --fail-under=80
  ```
when checking the package.  Your test files needs to start with the prefix "test_" to be run in Travis-CI.

* Check code style (no `flake8` warnings) with:

  ```bash
  $ pip install flake8
  $ flake8 pymcmcstat/path_to_module.py
  ```
  
 #### This guide was derived from the [PyMC3's guide to contributing](https://github.com/pymc-devs/pymc3/blob/master/CONTRIBUTING.md)
