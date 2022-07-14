from aiida.parsers import Parser

class Ld1Parser(Parser):
    """Parser for `Ld1Calculation` parse output to pseudo"""
    
    def parse(self, **kwargs):
        """Parse the contets of output of ld1 to pseudo files"""
        
        output_folder = self.retrieved
        
        with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
            stdout = handle.read()
        
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(stdout.encode('utf-8'))
            fpath = fp.name
            abi_parser = OncvParser(fpath)
            try:
                abi_parser.scan()
                crop_0_5_atan_logder_l1err = compute_crop_l1err(abi_parser.atan_logders, 0., 5)
            except:
                # not finish okay therefore not parsed
                # TODO re-check the following exit states, will be override by this one
                output_parameters = {}
                self.out('output_parameters', orm.Dict(dict=output_parameters))
                return self.exit_codes.get('ERROR_ABIPY_NOT_PARSED')
            
            results = abi_parser.get_results()
        
        output_parameters = {}
        
        output_parameters['crop_0_5_atan_logder_l1err'] = crop_0_5_atan_logder_l1err
        output_parameters['max_atan_logder_l1err'] = float(results['max_atan_logder_l1err'])
        output_parameters['max_ecut'] = float(results['max_ecut'])
        
        # Separate the input string into separate lines
        data_lines = stdout.split('\n') 
        logs = {'error': []}
        for count, line in enumerate(data_lines):
            
            # ERROR_PSPOT_HAS_NODE
            if 'pseudo wave function has node' in line:
                logs['error'].append('ERROR_PSPOT_HAS_NODE')
                
            if 'lschvkbb ERROR' in line:
                logs['error'].append('ERROR_LSCHVKBB')
            
            # line idx for PSP UPF part
            if 'Begin PSP_UPF' in line:
                start_idx = count
            
            if 'END_PSP' in line:
                end_idx = count
                
            # For configuration test results 
            # idx 0 always exist for the setting configuration
            if 'Test configuration'  in line:
                test_idx = line.strip()[-1]
                i = count
                
                while True:
                    if ('PSP excitation error' in data_lines[i]
                    or 'WARNING no output for configuration' in data_lines[i]):
                        end_count = i
                        break
                    
                    i += 1
                    
                test_ctx = data_lines[count+2:end_count+1]
                
                output_parameters[f'tc_{test_idx}'] = parse_configuration_test(test_idx, test_ctx)
                
        self.out('output_parameters', orm.Dict(dict=output_parameters))
                    
        if self.node.inputs.dump_psp.value:
            upf_lines = data_lines[start_idx+1:end_idx-1]
            upf_txt = '\n'.join(upf_lines)
            
            
            pseudo = UpfData.get_or_create(io.BytesIO(upf_txt.encode('utf-8')))
            self.out('output_pseudo', pseudo)
            
        for error_label in [
            'ERROR_PSPOT_HAS_NODE',
            'ERROR_LSCHVKBB',
        ]:
            if error_label in logs['error']:
                return self.exit_codes.get(error_label)
                