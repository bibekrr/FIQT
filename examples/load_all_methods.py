"""This example attempts to load all methods included in the toolkit, printing information for each."""

import fiqat


def main():
  # Get the MethodIds for each MethodType:
  method_id_by_method_type = {}
  for method_id in fiqat.registry.registry_data.keys():  # Iterate over entries for included methods.
    method_id: fiqat.MethodId  # Just for information.
    method_type = fiqat.registry.get_method_type_from_id(method_id)
    method_id_by_method_type.setdefault(method_type, []).append(method_id)

  # Try to load the methods categorized by MethodType and print information:
  for method_type in fiqat.MethodType:
    fiqat.term.cprint(method_type, 'cyan')
    for method_id in method_id_by_method_type.get(method_type, []):
      registry_entry: fiqat.RegistryEntry = fiqat.registry.registry_data[method_id]
      print(f"- Trying to load {fiqat.term.colored(method_id, 'cyan')}:")
      try:
        # Try to load the method:
        method_registry_entry: fiqat.MethodRegistryEntry = fiqat.registry.get_method(method_id)
        # Method could be loaded:
        fiqat.term.cprint(f"  - Status: {registry_entry['status']}", 'green')
        assert registry_entry['method_entry'] == method_registry_entry  # Just for information.
        fiqat.term.cprint(f"  - Default config: {method_registry_entry.get('default_config', {})}", 'green')
      except RuntimeError:
        # Method couldn't be loaded:
        fiqat.term.cprint(f"  - Status: {registry_entry['status']}", 'red')
        fiqat.term.cprint(f"  - Info: {registry_entry['info']}", 'red')
    print()


if __name__ == '__main__':
  main()
