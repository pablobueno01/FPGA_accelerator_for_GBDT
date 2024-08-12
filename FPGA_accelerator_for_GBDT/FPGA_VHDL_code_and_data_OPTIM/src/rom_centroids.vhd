-------------------------------------------------------------------------------
-- Synchronous ROM with generic memory and data sizes
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity rom is
    generic(ADDRESS_BITS: positive := 8;
            DATA_LENGTH:  positive := 13);
    port(-- Control signals
         Clk: in std_logic;
         Re:  in std_logic;
         
         -- Input signals
         Addr: in std_logic_vector (ADDRESS_BITS - 1 downto 0);
         
         -- Output
         Dout: out std_logic_vector (DATA_LENGTH - 1 downto 0));

end rom;

architecture Behavioral of rom is

    type MemoryBank is array(0 to 2**ADDRESS_BITS - 1)
                    of std_logic_vector(DATA_LENGTH - 1 downto 0);
    signal bank: MemoryBank;

begin

    bank <= (
        others => (others => '0')
    );

    process (Clk)
    begin
        if rising_edge(Clk) then
            if (Re = '1') then
                -- Read from Addr
                Dout <= bank(to_integer(unsigned(Addr)));
            else
                Dout <= (others => '0');
            end if;
        end if;
    end process;
end Behavioral;