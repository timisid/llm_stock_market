import streamlit as st

# Define each page as a separate Python file

page1 = st.Page("idx_explorer.py", title="IDX Explore (Mandatory)")

page2 = st.Page("idx_info.py", title="IDX Info (add_on)")

hideSidebarNav = True

# Create a navigation menu with each page as a separate tab
pg = st.navigation([page1, page2])

# Run the selected page
pg.run()