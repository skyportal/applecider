All the files and folders in this directory were moved over from the original AppleCider
repository. Generally speaking the code has not been altered from what it was prior
to the repository reorganization.

Because the code was not altered during the repository reorganization, it is likely
that any file that imports something from AppleCider will need to be fixed. Therefore
if you are pulling a module out of the archive, you should anticipate needing to
correct import problems.

Generally speaking to correct an import problem, all you will need to do is:
1) Ensure that what is being imported is already in the `.../src/applecider/...` directory
2) If the thing being imported isn't in that directory, pull it out of `_archive` too
3) Update `from AppleCider.foo.bar import baz` to something like `from applecider.foo.bar import baz`.
